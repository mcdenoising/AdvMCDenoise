import os
import math
from datetime import datetime
import numpy as np
import cv2
from torchvision.utils import make_grid
import random
import torch
import logging
import OpenEXR
import Imath


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False):
    '''set up logger'''
    l = logging.getLogger(logger_name)
    formatter = logging.Formatter(
        '%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s', datefmt='%y-%m-%d %H:%M:%S')
    log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        l.addHandler(sh)




def vray_tone_mapping(data, darkMultiplier=1.2, brightMultiplier=1.5):
    # gamma = 0.18;
    gamma = 1
    result = np.copy(data)
    brightness = np.sum(result * [0.114, 0.587, 0.299], axis=-1)
    result *= np.maximum(brightness * (1 - darkMultiplier) + darkMultiplier, 1.0)[:, :, np.newaxis]
    result *= np.minimum(brightness * (brightMultiplier - 1) + 1, brightMultiplier)[:, :, np.newaxis]
    result = 1 - np.exp(-result*gamma)
    return result   

def tonemap(matrix, gamma=2.2):
  return np.clip(matrix ** (1.0/gamma), 0, 1)

def tensor2exr(image_tensor, imtype=np.float32, normalize=True, ifExp=True):

    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2exr(image_tensor[i], imtype, normalize))
        return image_numpy
    # image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = image_tensor.numpy()  
    # image_numpy =  np.clip(image_numpy,0.0,1.0)

    if normalize:
        shape = image_numpy.shape
        if shape[0] == 10:
            image_numpy = image_numpy[:3,:,:]
        elif shape[2] == 10:
            image_numpy = image_numpy[:,:,:3]
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0   
        image_numpy = np.transpose(image_numpy, (1, 2, 0))    #CHANGED * 255.0
        #CHANGED invert the normalization properties.
        if ifExp:
            image_numpy = np.exp(image_numpy) - 1.0
        image_numpy = image_numpy

    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0))       
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)


def saveFeature(image_tensor, imtype=np.uint8):
    feature = image_tensor.numpy()
    feature = np.transpose(feature, (1, 2, 0))    #CHANGED * 255.0
    # feature= 255.0/(1+np.exp(-1*feature))
    feature = (feature - np.min(feature) )/(np.max(feature) - np.min(feature))
    feature = feature*255.0

    print(np.max(feature), np.min(feature))
    print(feature.shape)
    return feature.astype(imtype)



def tensor2img(image_tensor, imtype=np.uint8, normalize=True, ifExp=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2img(image_tensor[i], imtype, normalize))
        return image_numpy
    # image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = image_tensor.numpy()  

    if normalize:
        shape = image_numpy.shape
        if shape[0] >3 and shape[0]<30:
            image_numpy = image_numpy[:3,:,:]
        elif shape[2] >3 and shape[2]<30:
            image_numpy = image_numpy[:,:,:3]
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0   
        image_numpy = np.transpose(image_numpy, (1, 2, 0))    #CHANGED * 255.0
        #CHANGED invert the normalization properties.
        if ifExp:
            image_numpy = np.exp(image_numpy) - 1.0
        image_numpy = tonemap(image_numpy)*255.0
        # image_numpy = vray_tone_mapping(image_numpy)*255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0      
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)


def saveEXRfromMatrix(file,data,size):
    output = OpenEXR.OutputFile(file, OpenEXR.Header(*size))
    data = np.moveaxis(data, -1, 0)
    output.writePixels({
        "R": data[0].tostring(),
        "G": data[1].tostring(),
        "B": data[2].tostring(),
        # "A": data[3].tostring()
    })

def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)


####################
# metric
####################


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        # return float('inf')
        return 0.0
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def mrse(img1, img2):
    num = (img1 - img2)**2
    denom = img2**2 + 1.0e-2
    relMse = np.divide(num, denom)
    relMseMean = 0.5*np.mean(relMse)
    return relMseMean

def calculate_mrse(img1, img2):
    '''calculate MRSE
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    return mrse(img1, img2)
