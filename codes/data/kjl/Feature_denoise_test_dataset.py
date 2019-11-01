import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util

TRAIN_DATA_NUM = 900
class FeatureDenoiseDataset(data.Dataset):
    '''
    Read in the reference, noisy image and auxiliary features 
    '''
    def __init__(self, opt):
        super(FeatureDenoiseDataset, self).__init__()
        self.opt = opt
        self.paths_NOISY = None
        self.paths_GT = None

        self.FEATURE_DIR = os.path.join(opt['dataroot_NOISY'])
        self.REF_DIR = os.path.join(opt['dataroot_GT'])
        print(self.REF_DIR)

        # input A
        self.paths_NOISY = sorted(util.get_dirs_in_dir(self.FEATURE_DIR))
        self.paths_GT = []
        for path_NOISY in self.paths_NOISY:
            self.paths_GT.append(os.path.join(self.REF_DIR, path_NOISY+"xr.exr"))
        #input B
        # self.paths_GT = sorted(util.make_dataset(self.REF_DIR))
       
        # self.paths_NOISY = self.paths_NOISY[TRAIN_DATA_NUM :]
        # self.paths_GT = self.paths_GT[TRAIN_DATA_NUM :]
        print(len(self.paths_GT))
        
        
        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_NOISY and self.paths_GT:
            assert len(self.paths_NOISY) == len(self.paths_GT), \
                'GT and NOISY datasets have different number of images - {}, {}.'.format(\
                len(self.paths_NOISY), len(self.paths_GT))


    def __getitem__(self, index):
        GT_path, NOISY_path = None, None
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]
        # img_GT = util.read_img(self.GT_env, GT_path)
        # for HDRs
        img_GT = util.load_reference_mat("", GT_path)

        # get NOISY image
        if self.paths_NOISY:
            NOISY_path = self.paths_NOISY[index]
            all_feature = util.load_feature_mat(self.FEATURE_DIR, NOISY_path)  #util.load_feature_mat_complete_tungsten
            img_NOISY = all_feature[:,:,:3]
            features = all_feature[:,:,3:]
        # do the cropping
        H, W, _ = img_GT.shape
        # x = random.randint(0, np.maximum(0, W - GT_size))
        # y = random.randint(0, np.maximum(0, H - GT_size))
        x = 0
        y = 0
        img_GT = util._crop(img_GT, (y,x), GT_size)
        img_NOISY = util._crop(img_NOISY, (y,x), GT_size)
        features = util._crop(features, (y,x), GT_size)

        if self.opt['phase'] == 'train':
            # augmentation - flip, rotate
            img_NOISY, img_GT, features = util.augment([img_NOISY, img_GT, features], self.opt['use_flip'], \
                self.opt['use_rot'])

        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_NOISY = torch.from_numpy(np.ascontiguousarray(np.transpose(img_NOISY, (2, 0, 1)))).float()
        features = torch.from_numpy(np.ascontiguousarray(np.transpose(features, (2, 0, 1)))).float()

        return {
        'diffuse_in': img_NOISY, 
        'diffuse_ref': img_GT, 
        "seg":features, 
        "category": 1, 
        'NOISY_path': NOISY_path, 
        'GT_path': GT_path,
        "x_offset": 512,
        "y_offset": 512}


    def __len__(self):
        return len(self.paths_GT)
