import os
import math
import pickle
import random
import numpy as np
# import lmdb
import torch
import cv2
import logging
from os import listdir
from os.path import isfile, join, isdir,exists
import OpenEXR
import Imath
import data.util_exr as exr_utils

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', ".exr"]

####################
# Files & IO
####################


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def get_image_paths(data_type, dataroot):
    env, paths = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            env, paths = _get_paths_from_lmdb(dataroot)
        elif data_type == 'img':
            paths = sorted(_get_paths_from_images(dataroot))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return env, paths


def _read_lmdb_img(env, path):
#     with env.begin(write=False) as txn:
#         buf = txn.get(path.encode('ascii'))
#         buf_meta = txn.get((path + '.meta').encode('ascii')).decode('ascii')
#     img_flat = np.frombuffer(buf, dtype=np.uint8)
#     H, W, C = [int(s) for s in buf_meta.split(',')]
#     img = img_flat.reshape(H, W, C)
    return None
    # return img


def read_img(env, path):
    # read image by cv2 or from lmdb
    # return: Numpy float32, HWC, BGR, [0,1]
    if env is None:  # img
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    else:
        img = _read_lmdb_img(env, path)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images

####################
# image processing
# process on numpy image
####################


def augment(img_list, hflip=True, rot=True):
    # horizontal flip OR rotate
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def rgb2ycbcr(img, only_y=True):
    '''same as matlab rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786],
                              [24.966, 112.0, -18.214]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def ycbcr2rgb(img):
    '''same as matlab ycbcr2rgb
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    rlt = np.matmul(img, [[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],
                          [0.00625893, -0.00318811, 0]]) * 255.0 + [-222.921, 135.576, -276.836]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def modcrop(img_in, scale):
    # img_in: Numpy, HWC or HW
    img = np.copy(img_in)
    if img.ndim == 2:
        H, W = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r]
    elif img.ndim == 3:
        H, W, C = img.shape
        H_r, W_r = H % scale, W % scale
        img = img[:H - H_r, :W - W_r, :]
    else:
        raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
    return img



def _crop(img, pos, size):
    ow, oh = img.shape[0], img.shape[1]  
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):        
        # return img.crop((x1, y1, x1 + tw, y1 + th)) #CHANGED
        return img[x1:(x1 + tw), y1:(y1 + th), :]
    return img

####################
# Functions
####################


# matlab 'imresize' function, now only support 'bicubic'
def cubic(x):
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5*absx3 - 2.5*absx2 + 1) * ((absx <= 1).type_as(absx)) + \
        (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * (((absx > 1)*(absx <= 2)).type_as(absx))


def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    if (scale < 1) and (antialiasing):
        # Use a modified kernel to simultaneously interpolate and antialias- larger kernel width
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5+scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    P = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, P) + torch.linspace(0, P - 1, P).view(
        1, P).expand(out_length, P)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, P) - indices
    # apply cubic kernel
    if (scale < 1) and (antialiasing):
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)
    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, P)

    # If a column in weights is all zero, get rid of it. only consider the first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, P - 2)
        weights = weights.narrow(1, 1, P - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, P - 2)
        weights = weights.narrow(1, 0, P - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


def imresize(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: CHW RGB [0,1]
    # output: CHW RGB [0,1] w/o round

    in_C, in_H, in_W = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_C, in_H + sym_len_Hs + sym_len_He, in_W)
    img_aug.narrow(1, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:, :sym_len_Hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_He:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_C, out_H, in_W)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[0, i, :] = img_aug[0, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        out_1[1, i, :] = img_aug[1, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])
        out_1[2, i, :] = img_aug[2, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_C, out_H, in_W + sym_len_Ws + sym_len_We)
    out_1_aug.narrow(2, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_Ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_We:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_C, out_H, out_W)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[0, :, i] = out_1_aug[0, :, idx:idx + kernel_width].mv(weights_W[i])
        out_2[1, :, i] = out_1_aug[1, :, idx:idx + kernel_width].mv(weights_W[i])
        out_2[2, :, i] = out_1_aug[2, :, idx:idx + kernel_width].mv(weights_W[i])

    return out_2


def imresize_np(img, scale, antialiasing=True):
    # Now the scale should be the same for H and W
    # input: img: Numpy, HWC BGR [0,1]
    # output: HWC BGR [0,1] w/o round
    img = torch.from_numpy(img)

    in_H, in_W, in_C = img.size()
    out_C, out_H, out_W = in_C, math.ceil(in_H * scale), math.ceil(in_W * scale)
    kernel_width = 4
    kernel = 'cubic'

    # Return the desired dimension order for performing the resize.  The
    # strategy is to perform the resize first along the dimension with the
    # smallest scale factor.
    # Now we do not support this.

    # get weights and indices
    weights_H, indices_H, sym_len_Hs, sym_len_He = calculate_weights_indices(
        in_H, out_H, scale, kernel, kernel_width, antialiasing)
    weights_W, indices_W, sym_len_Ws, sym_len_We = calculate_weights_indices(
        in_W, out_W, scale, kernel, kernel_width, antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_H + sym_len_Hs + sym_len_He, in_W, in_C)
    img_aug.narrow(0, sym_len_Hs, in_H).copy_(img)

    sym_patch = img[:sym_len_Hs, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, 0, sym_len_Hs).copy_(sym_patch_inv)

    sym_patch = img[-sym_len_He:, :, :]
    inv_idx = torch.arange(sym_patch.size(0) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(0, inv_idx)
    img_aug.narrow(0, sym_len_Hs + in_H, sym_len_He).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(out_H, in_W, in_C)
    kernel_width = weights_H.size(1)
    for i in range(out_H):
        idx = int(indices_H[i][0])
        out_1[i, :, 0] = img_aug[idx:idx + kernel_width, :, 0].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 1] = img_aug[idx:idx + kernel_width, :, 1].transpose(0, 1).mv(weights_H[i])
        out_1[i, :, 2] = img_aug[idx:idx + kernel_width, :, 2].transpose(0, 1).mv(weights_H[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(out_H, in_W + sym_len_Ws + sym_len_We, in_C)
    out_1_aug.narrow(1, sym_len_Ws, in_W).copy_(out_1)

    sym_patch = out_1[:, :sym_len_Ws, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, 0, sym_len_Ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, -sym_len_We:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    out_1_aug.narrow(1, sym_len_Ws + in_W, sym_len_We).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(out_H, out_W, in_C)
    kernel_width = weights_W.size(1)
    for i in range(out_W):
        idx = int(indices_W[i][0])
        out_2[:, i, 0] = out_1_aug[:, idx:idx + kernel_width, 0].mv(weights_W[i])
        out_2[:, i, 1] = out_1_aug[:, idx:idx + kernel_width, 1].mv(weights_W[i])
        out_2[:, i, 2] = out_1_aug[:, idx:idx + kernel_width, 2].mv(weights_W[i])

    return out_2.numpy()


if __name__ == '__main__':
    # test imresize function
    # read images
    img = cv2.imread('test.png')
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    # imresize
    scale = 1 / 4
    import time
    total_time = 0
    for i in range(10):
        start_time = time.time()
        rlt = imresize(img, scale, antialiasing=True)
        use_time = time.time() - start_time
        total_time += use_time
    print('average time: {}'.format(total_time / 10))

    import torchvision.utils
    torchvision.utils.save_image(
        (rlt * 255).round() / 255, 'rlt.png', nrow=1, padding=0, normalize=False)


# utilities for denoising HDRs

def get_files_in_dir(dir_path):
    names = []
    for f in listdir(dir_path):
        if isfile(join(dir_path, f)):
            names.append(f.split(".")[0].split("_")[0])
    return names

def get_dirs_in_dir(dir_path):
    names = []
    for f in listdir(dir_path):
        if isdir(join(dir_path, f)):
            names.append(f)
    return names


def loadEXR2matrix(path,channel=3):
    image = OpenEXR.InputFile(path)
    dataWindow = image.header()['dataWindow']
    size = (dataWindow.max.x - dataWindow.min.x + 1, dataWindow.max.y - dataWindow.min.y + 1)
    HALF = Imath.PixelType(Imath.PixelType.HALF)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    if channel == 3:
        data = np.array([np.fromstring(image.channel(c, FLOAT), dtype=np.float32) for c in 'BGR'])
    elif channel == 4:
        data = np.array([np.fromstring(image.channel(c, FLOAT), dtype=np.float32) for c in 'BGRA'])
    data = np.moveaxis(data, 0, -1)
    data = data.reshape(size[1], size[0],channel)
    return data


TRAIN_FEATURE_CHANNELS_DIC = {"color": 3, "normal": 3, "texture": 3, "depth": 1, "variances":3, "roughness": 1}

CURRENT_FEATURE_NAME_LIST = ["color", "normal", "depth", "texture"] #

def load_feature_mat_patch(path):

    color_path = join(path +"_color.exr")
    normal_path = join(path + "_normal.exr")
    depth_path = join(path + "_depth.exr")
    texture_path = join(path + "_texture.exr")
    

    res_tuple = ()
    
    if "color" in CURRENT_FEATURE_NAME_LIST:
        color = loadEXR2matrix(color_path)
        color = np.log(color+1)
        res_tuple = res_tuple + (color,) 

    if "normal" in CURRENT_FEATURE_NAME_LIST:
        normal = loadEXR2matrix(normal_path)
        normal = np.nan_to_num(normal)
        # normal = (normal + 1.0)*0.5
        # normal = np.maximum(np.minimum(normal,1.0),0.0)
        res_tuple = res_tuple + (normal, )
    
    if "depth" in CURRENT_FEATURE_NAME_LIST:
        depth = loadEXR2matrix(depth_path)[:,:,:1]
        # depth  = depth/np.max(depth)
        res_tuple = res_tuple + (depth, )

    if "texture" in CURRENT_FEATURE_NAME_LIST:
        texture= loadEXR2matrix(texture_path)
        res_tuple = res_tuple + (texture, )  

    mat = np.concatenate(res_tuple, axis=2)
    return mat

def load_feature_mat(path, ID):

    ID = str(ID)
    color_path = join(path,ID, "color.exr")
    normal_path = join(path,ID, "normal.exr")
    depth_path = join(path, ID, "depth.exr")
    texture_path = join(path, ID, "texture.exr")
    

    res_tuple = ()
    
    if "color" in CURRENT_FEATURE_NAME_LIST:
        color = loadEXR2matrix(color_path)
        color = np.log(color+1)
        res_tuple = res_tuple + (color,) 

    if "normal" in CURRENT_FEATURE_NAME_LIST:
        normal = loadEXR2matrix(normal_path)
        normal = np.nan_to_num(normal)
        # normal = (normal + 1.0)*0.5
        # normal = np.maximum(np.minimum(normal,1.0),0.0)
        res_tuple = res_tuple + (normal, )
    
    if "depth" in CURRENT_FEATURE_NAME_LIST:
        depth = loadEXR2matrix(depth_path)[:,:,:1]
        # depth  = depth/np.max(depth)
        res_tuple = res_tuple + (depth, )

    if "texture" in CURRENT_FEATURE_NAME_LIST:
        texture= loadEXR2matrix(texture_path)
        res_tuple = res_tuple + (texture, )  

    mat = np.concatenate(res_tuple, axis=2)
    return mat

def load_feature_mat_patch_shading(path):

    color_path = join(path +"_color.exr")
    normal_path = join(path + "_normal.exr")
    depth_path = join(path + "_depth.exr")
    texture_path = join(path + "_albedo.exr")
    

    res_tuple = ()
    
    if "color" in CURRENT_FEATURE_NAME_LIST:
        color = loadEXR2matrix(color_path)

    if "normal" in CURRENT_FEATURE_NAME_LIST:
        normal = loadEXR2matrix(normal_path)
        normal = np.nan_to_num(normal)
    
    if "depth" in CURRENT_FEATURE_NAME_LIST:
        depth = loadEXR2matrix(depth_path)[:,:,:1]

    if "texture" in CURRENT_FEATURE_NAME_LIST:
        texture= loadEXR2matrix(texture_path)

    color = color/(texture+ 1e-3)
    color = np.log(color+1)
    res_tuple = (color, normal, depth)     
    mat = np.concatenate(res_tuple, axis=2)
    return mat


def load_reference_mat(path_ref, scene_name):
    file_full_path = join(path_ref, scene_name)
    color = loadEXR2matrix(file_full_path)    
    color = np.log(color+1)
    return color

def load_reference_tungsten_mat_shading(path_ref, scene_name, path_feature=None):
    file_full_path = join(path_ref, scene_name)
    color = loadEXR2matrix(file_full_path)    
    if path_feature != None:
        texture_path = join(path_feature.rsplit(".",1)[0] + "_albedo.exr")
        texture= loadEXR2matrix(texture_path)
        color = color/ (texture + 1e-3)
    color = np.log(color+1)
    return color
# Calculate log transform (with an offset to map zero to zero)
def LogTransform(data):
    assert(np.sum(data < 0) == 0)
    return np.log(data + 1.0)

def loadEXR2matrixYChannel(path):
    image = OpenEXR.InputFile(path)
    dataWindow = image.header()['dataWindow']
    size = (dataWindow.max.x - dataWindow.min.x + 1, dataWindow.max.y - dataWindow.min.y + 1)
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    data = np.array([np.fromstring(image.channel(c, FLOAT), dtype=np.float32) for c in 'Y'])
    data = np.moveaxis(data, 0, -1)
    data = data.reshape(size[1], size[0], 1) #(600,800,3)
    return data 



def load_feature_mat_complete_tungsten_joint(path, FEATURE_LIST):

    prefix = path.split(".")[0]

    color_path = prefix + "_color.exr"
    variance_path = prefix + "_variance.exr"
    normal_path = prefix + "_normal.exr"
    depth_path = prefix + "_depth.exr"
    texture_path = prefix + "_albedo.exr"
    # visibility_path = prefix + "_visibility.exr"

    diffuse_path = prefix + "_diffuse.exr"
    specular_path = prefix + "_specular.exr"
    try:
        # inFile = exr_utils.open(color_path)
        # color = inFile.get_all()["default"]
        inFile = exr_utils.open(normal_path)
        normal = inFile.get_all()["default"]
        inFile = exr_utils.open(depth_path)
        depth = inFile.get_all()["default"]
        inFile = exr_utils.open(texture_path)
        texture = inFile.get_all()["default"]

        # inFile = exr_utils.open(visibility_path)
        # visibility = inFile.get_all()["default"]

        inFile = exr_utils.open(diffuse_path)
        diffuse = inFile.get_all()["default"]
        inFile = exr_utils.open(specular_path)
        specular = inFile.get_all()["default"]

    except Exception:
        print("Exception!!!!!"+prefix)
        return None, None, None,None

    # variance = CalcRelVar( (1+ color.copy()) , variance, False, False, True )
    # color[color < 0.0] = 0.0
    # color = LogTransform(color)
    diffuse[diffuse < 0.0] = 0.0
    diffuse = diffuse / (texture + 0.00316)
    diffuse = LogTransform(diffuse)
    specular[specular < 0.0] = 0.0
    specular = LogTransform(specular)
    normal = np.nan_to_num(normal)
    if "specular" in FEATURE_LIST:
        normal = (normal + 1.0)*0.5
        normal = np.maximum(np.minimum(normal,1.0),0.0)
    # Normalize current frame depth to [0,1]
    maxDepth = np.max(depth)
    if maxDepth != 0:
        depth /= maxDepth

    #feature shut off
    # normal.fill(0.0)
    # # visibility.fill(0.0)
    # depth.fill(0.0)
    # texture.fill(0.0)    

    # texture = np.clip(texture,0.0,1.0)
    
    # feautres = np.concatenate((variance,  normal, depth, texture, visibility), axis=2)    
    feautres = np.concatenate((normal, depth, texture), axis=2) 

    return diffuse, diffuse, specular, feautres
    # return np.concatenate((color, normal, depth, texture), axis=2)


def load_feature_mat_complete_tungsten(input_dir, scene_name):

    color_path = join(input_dir, scene_name, "color.exr")
    normal_path = join(input_dir, scene_name, "normal.exr")
    texture_path = join(input_dir, scene_name, "albedo.exr")
    depth_path = join(input_dir, scene_name, "depth.exr")

    texture = loadEXR2matrix(texture_path)
    res_tuple = ()

    if "color" in CURRENT_FEATURE_NAME_LIST:
        color = loadEXR2matrix(color_path)
        color = np.log(color+1)
        res_tuple = res_tuple + (color,)

    if "normal" in CURRENT_FEATURE_NAME_LIST:
        normal = loadEXR2matrix(normal_path)
        normal = np.nan_to_num(normal)
        normal = (normal + 1.0)*0.5
        normal = np.maximum(np.minimum(normal,1.0),0.0)
        res_tuple = res_tuple + (normal,)

    if "depth" in CURRENT_FEATURE_NAME_LIST:
        depth = loadEXR2matrixYChannel(depth_path)
        depth  = depth/np.max(depth)
        res_tuple = res_tuple + (depth, )

    if "texture" in CURRENT_FEATURE_NAME_LIST:
        texture= loadEXR2matrix(texture_path)
        texture = np.clip(texture,0.0,1.0)
        res_tuple = res_tuple + (texture, )      

    mat = np.concatenate(res_tuple, axis=2)
    return mat

