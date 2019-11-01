import logging
logger = logging.getLogger('base')


def create_model(opt):
    model = opt['model']
    
    if model == "resnet":
        from .Seperate_Denoise_Resnet_model import Seperate_Denoise_Resnet_Model as M 
    elif model == "seperate_cfm_gan":
        from .Seperate_Denoise_CFM_model import Seperate_Denoise_CFM_Model as M
    else:
        raise NotImplementedError('Model [{:s}] not recognized.'.format(model))
    m = M(opt)
    logger.info('Model [{:s}] is created.'.format(m.__class__.__name__))
    return m
