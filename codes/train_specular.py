import os.path
import sys
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging

import torch

import scripts.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

    # train from scratch OR resume training
    if opt['path']['resume_state']:  # resuming training
        resume_state = torch.load(opt['path']['resume_state'])
        
    else:  # training from scratch
        resume_state = None
        util.mkdir_and_rename(opt['path']['experiments_root'])  # rename old folder if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                     and 'pretrain_model' not in key and 'resume' not in key))

    # config loggers. Before it, the log will not work
    util.setup_logger(None, opt['path']['log'], 'train', level=logging.INFO, screen=True)
    util.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')

    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))
        option.check_resume(opt)  # check resume options

    logger.info(option.dict2str(opt))
    # tensorboard logger
    if opt['use_tb_logger'] and 'debug' not in opt['name']:
        from tensorboardX import SummaryWriter
        tb_logger = SummaryWriter(log_dir='../tb_logger/' + opt['name'])

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    logger.info('Random seed: {}'.format(seed))
    util.set_random_seed(seed)

    torch.backends.cudnn.benckmark = True
    # torch.backends.cudnn.deterministic = True

    # create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                len(train_set), train_size))
            total_iters = int(opt['train']['niter'])
            total_epochs = int(math.ceil(total_iters / train_size))
            logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                total_epochs, total_iters))
            train_loader = create_dataloader(train_set, dataset_opt)
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            logger.info('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'],
                                                                      len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    # create model
    model = create_model(opt)

    # resume training
    if resume_state:
        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    # training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs):
        for _, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            # update learning rate
            model.update_learning_rate()

            # training
            model.feed_data_specular(train_data)
            model.optimize_parameters(current_step)

            # log
            if current_step % opt['logger']['print_freq'] == 0:
                logs = model.get_current_log()
                message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
                    epoch, current_step, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += '{:s}: {:.4e} '.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar(k, v, current_step)
                logger.info(message)

            # validation
            if current_step % opt['train']['val_freq'] == 0:
                avg_psnr = 0.0
                avg_ssim = 0.0
                avg_mrse = 0.0
                idx = 0
                for val_data in val_loader:
                    idx += 1
                    img_name = os.path.splitext(os.path.basename(val_data['NOISY_path'][0]))[0]
                    img_dir = opt['path']['val_images']  #os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)

                    model.feed_data_specular(val_data)
                    model.test()
                    if opt["image_type"] == "exr":
                        y = val_data["x_offset"]
                        x = val_data["y_offset"]                    
                    visuals = model.get_current_visuals()
                    avg_mrse += util.calculate_mrse(visuals["DENOISED"].numpy(), visuals["GT"].numpy())
                    lr_img = util.tensor2img(visuals['NOISY'])
                    sr_img = util.tensor2img(visuals['DENOISED'])  # uint8
                    gt_img = util.tensor2img(visuals['GT'])  # uint8

############################################################################################## 
                    # sr_img = util.tensor2img(visuals['DENOISED'])  # uint8
                    # lr_img = util.tensor2img(visuals['NOISY'])
                    # gt_img = util.tensor2img(visuals['GT'])  # uint8

                    # if opt["image_type"] == "exr":
                    #     sr_img = sr_img[y:1280-y, x:1280-x, :]
                    #     lr_img = lr_img[y:1280-y, x:1280-x, :]
                    #     gt_img = gt_img[y:1280-y, x:1280-x, :]


##############################################################################################


                    # Save DENOISED images for reference
                    save_DENOISED_img_path = os.path.join(img_dir, '{:s}_{:d}_1denoised.png'.format(img_name, current_step))
                    save_NOISY_img_path = os.path.join(img_dir, '{:s}_{:d}_0noisy.png'.format(img_name, current_step))
                    save_GT_img_path = os.path.join(img_dir, '{:s}_{:d}_2gt.png'.format(img_name, current_step))
                    # if current_step % 10000 == 0 :#and idx%100 ==0:
                    #     util.save_img(sr_img, save_DENOISED_img_path)
                    #     util.save_img(lr_img, save_NOISY_img_path)
                    #     util.save_img(gt_img, save_GT_img_path)

                    # calculate PSNR
                    # crop_size = opt['scale']
                    gt_img = gt_img #/ 255.
                    sr_img = sr_img #/ 255.
                    # cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    # cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size, :]
                    # avg_psnr += util.calculate_psnr(sr_img * 255, gt_img * 255)
                    avg_psnr += util.calculate_psnr(sr_img , gt_img )
                    avg_ssim += util.calculate_ssim(sr_img , gt_img)

##############################################################################################

                    if opt["image_type"] == "exr" and  current_step %10000 == 0:
                        sr_exr = util.tensor2exr(visuals['DENOISED'])  # uint8
                        lr_exr = util.tensor2exr(visuals['NOISY'])
                        gt_exr = util.tensor2exr(visuals['GT'])  # uint8

                        # sr_exr = sr_exr[y:1280-y, x:1280-x, :]
                        # lr_exr = lr_exr[y:1280-y, x:1280-x, :]
                        # gt_exr = gt_exr[y:1280-y, x:1280-x, :]
                        save_DENOISED_img_path = os.path.join(img_dir, '{:s}_{:d}_1denoised.exr'.format(img_name, current_step))
                        save_NOISY_img_path = os.path.join(img_dir, '{:s}_{:d}_0noisy.exr'.format(img_name, current_step))
                        save_GT_img_path = os.path.join(img_dir, '{:s}_{:d}_2gt.exr'.format(img_name, current_step)) 
                    
                        util.saveEXRfromMatrix(save_DENOISED_img_path, sr_exr, (x, y)) 
                        util.saveEXRfromMatrix(save_NOISY_img_path, lr_exr, (x, y))  
                        util.saveEXRfromMatrix(save_GT_img_path, gt_exr, (x, y))

##############################################################################################

                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                avg_mrse = avg_mrse / idx

                # log
                logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
                logger.info('# Validation # MRSE: {:.4e}'.format(avg_mrse))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e} ssim: {:.4e} mrse:  {:.4e}'.format(
                    epoch, current_step, avg_psnr, avg_ssim, avg_mrse))
                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('ssim', avg_ssim, current_step)
                    tb_logger.add_scalar('mrse', avg_mrse, current_step)

            # save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(current_step)
                model.save_training_state(epoch, current_step)

    logger.info('Saving the final model.')
    model.save('latest')
    logger.info('End of training.')


if __name__ == '__main__':
    main()
