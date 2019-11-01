'''create dataset and dataloader'''
import logging
import torch.utils.data


def create_dataloader(dataset, dataset_opt):
    '''create dataloader '''
    phase = dataset_opt['phase']
    if phase == 'train':
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=dataset_opt['batch_size'],
            shuffle=dataset_opt['use_shuffle'],
            num_workers=dataset_opt['n_workers'],
            drop_last=True,
            pin_memory=True)
    else:
        return torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


def create_dataset(dataset_opt):
    '''create dataset'''
    mode = dataset_opt['mode']
    if mode == "Feature_denoise":
        from data.kjl.Feature_denoise_dataset import FeatureDenoiseDataset as D       
    elif mode == "Feature_denoise_test":
        from data.kjl.Feature_denoise_test_dataset import FeatureDenoiseDataset as D
    elif mode == "Disney_feature_denoise_diffuse":
        from data.Multi_feature_denoise_dataset_diffuse import FeatureDenoiseDataset as D
    elif mode == "Disney_feature_denoise_specular":
        from data.Multi_feature_denoise_dataset_specular import FeatureDenoiseDataset as D    
    elif mode == "Tungste_testset_joint":
        from .Tungsten_testset_joint import TungstenTestset as D    
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
    dataset = D(dataset_opt)
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
