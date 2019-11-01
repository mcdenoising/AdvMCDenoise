import os
import sys
import logging
import time
import argparse
import numpy as np
from collections import OrderedDict

import scripts.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model

# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
opt = option.dict_to_nonedict(opt)

util.setup_logger(None, opt['path']['log'], 'test.log', level=logging.INFO, screen=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))
# Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# Create model
model = create_model(opt)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []

    for data in test_loader:
        need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True
        # need_GT = True

        model.feed_data_specular(data, need_GT=need_GT)
        if opt["image_type"] == "exr":
            y = data["x_offset"]
            x = data["y_offset"]
        img_path = data['NOISY_path'][0]
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        start = time.time()
        model.test()  # test
        end = time.time()
        print("Time elapsed... %f "%(end - start))
        visuals = model.get_current_visuals(need_GT=need_GT)

        denoised_img = util.tensor2img(visuals['DENOISED'])  # uint8
        noisy_img = util.tensor2img(visuals['NOISY'])
        gt_img = util.tensor2img(visuals['GT'])  # uint8

        # save images
        suffix = opt['suffix']
        if suffix ==None:
            suffix = ""
            
        save_DENOISED_img_path = os.path.join(dataset_dir, img_name + suffix + '_1denoised.png')
        save_NOISY_img_path = os.path.join(dataset_dir, img_name + suffix + '_0noisy.png')
        save_GT_img_path = os.path.join(dataset_dir, img_name + suffix + '_2gt.png')
  
        
        # calculate PSNR and SSIM
        if need_GT:
            # gt_img = util.tensor2img(visuals['GT'])
            gt_img = gt_img / 255.
            denoised_img = denoised_img / 255.

            crop_border = test_loader.dataset.opt['scale']
            cropped_denoised_img = denoised_img#[crop_border:-crop_border, crop_border:-crop_border, :]
            cropped_gt_img = gt_img#[crop_border:-crop_border, crop_border:-crop_border, :]

            psnr = util.calculate_psnr(cropped_denoised_img * 255, cropped_gt_img * 255)
            ssim = util.calculate_ssim(cropped_denoised_img * 255, cropped_gt_img * 255)
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)

            if gt_img.shape[2] == 3:  # RGB image
                denoised_img_y = bgr2ycbcr(denoised_img, only_y=True)
                gt_img_y = bgr2ycbcr(gt_img, only_y=True)
                cropped_denoised_img_y = denoised_img_y[crop_border:-crop_border, crop_border:-crop_border]
                cropped_gt_img_y = gt_img_y[crop_border:-crop_border, crop_border:-crop_border]
                psnr_y = util.calculate_psnr(cropped_denoised_img_y * 255, cropped_gt_img_y * 255)
                ssim_y = util.calculate_ssim(cropped_denoised_img_y * 255, cropped_gt_img_y * 255)
                test_results['psnr_y'].append(psnr_y)
                test_results['ssim_y'].append(ssim_y)
                logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.'\
                    .format(img_name, psnr, ssim, psnr_y, ssim_y))
            else:
                logger.info('{:20s} - PSNR: {:.6f} dB; SSIM: {:.6f}.'.format(img_name, psnr, ssim))
        else:
            logger.info(img_name)


        if opt["image_type"] == "exr":
            denoised_exr = util.tensor2exr(visuals['DENOISED'])  # uint8
            noisy_exr = util.tensor2exr(visuals['NOISY'])
            gt_exr = util.tensor2exr(visuals['GT'])  # uint8

            save_DENOISED_img_path = os.path.join(dataset_dir, img_name + suffix + '_1denoised.exr')
            save_NOISY_img_path = os.path.join(dataset_dir, img_name + suffix + '_0noisy.exr')
            save_GT_img_path = os.path.join(dataset_dir, img_name + suffix + '_2gt.exr') 
  
            util.saveEXRfromMatrix(save_DENOISED_img_path, denoised_exr, (x, y)) 
            util.saveEXRfromMatrix(save_NOISY_img_path, noisy_exr, (x, y))  
            util.saveEXRfromMatrix(save_GT_img_path, gt_exr, (x, y))

    if need_GT:  # metrics
        # Average PSNR/SSIM results
        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info('----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'\
                .format(test_set_name, ave_psnr, ave_ssim))
        # if test_results['psnr_y'] and test_results['ssim_y']:
        #     ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
        #     ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
        #     logger.info('----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'\
        #         .format(ave_psnr_y, ave_ssim_y))
