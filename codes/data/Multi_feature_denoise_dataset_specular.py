import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util
import data.util_disney as util_disney


# for Disney dataset
SCENE_NAME_LIST = ["bathroom2", "house", "spaceship","staircase", "car2","room2", "room3","classroom"] #  

class FeatureDenoiseDataset(data.Dataset):
    def __init__(self, opt):
        super(FeatureDenoiseDataset, self).__init__()
        self.opt = opt
        self.paths_GT = []
        self.paths_NOISY = []

        for scece_name in SCENE_NAME_LIST:
            scene_dir = os.path.join(opt['dataroot_GT'], scece_name)
            distinct_prefix = util_disney.get_distinct_prefix(scene_dir)
            self.paths_GT.extend([os.path.join(opt["dataroot_GT"], scece_name, prefix+"-32768spp.exr") for prefix in distinct_prefix if os.path.exists(os.path.join(opt["dataroot_GT"], scece_name, prefix+"-32768spp_specular.exr"))])
            self.paths_NOISY.extend([os.path.join(opt["dataroot_NOISY"], scece_name, prefix+"-00032spp.exr") for prefix in distinct_prefix if os.path.exists(os.path.join(opt["dataroot_GT"], scece_name, prefix+"-32768spp_specular.exr"))])

        self.paths_GT = sorted(self.paths_GT)    
        self.paths_NOISY = sorted(self.paths_NOISY)
        combined = list(zip(self.paths_GT, self.paths_NOISY))
        random.shuffle(combined)
        self.paths_GT , self.paths_NOISY = zip(*combined)
        scene_num = len(self.paths_GT)
        TRAIN_DATA_NUM = int(scene_num*0.95)
        # TRAIN_DATA_NUM = scene_num - 1000 #int(scene_num*0.95)
        if self.opt['phase'] == 'train':
            self.paths_NOISY = self.paths_NOISY[:TRAIN_DATA_NUM]
            self.paths_GT = self.paths_GT[:TRAIN_DATA_NUM]
        elif self.opt['phase'] == 'val':
            self.paths_NOISY = self.paths_NOISY[TRAIN_DATA_NUM :]
            self.paths_GT = self.paths_GT[TRAIN_DATA_NUM :]
            # self.paths_NOISY = self.paths_NOISY[scene_num - 1000 : scene_num]
            # self.paths_GT = self.paths_GT[scene_num - 1000 : scene_num]


        print("[INFO] total number of training_set %d"% len(self.paths_GT))    
        

    def __getitem__(self, index):
        GT_path, NOISY_path = None, None
        GT_size = self.opt['GT_size']

        GT_path = self.paths_GT[index]
        specular_ref = util_disney.loadDisneyEXR_multi_ref_shading(GT_path, self.opt["feature"]+["specular"])
       
        NOISY_path = self.paths_NOISY[index]
        specular_in, features = util_disney.loadDisneyEXR_feature_shading(NOISY_path, self.opt["feature"]+["specular"])

        if self.opt['phase'] == 'train':
            # augmentation - flip, rotate
            specular_ref, specular_in, features= util.augment([ specular_ref, specular_in, features], self.opt['use_flip'], \
                self.opt['use_rot'])
            

        
        features = torch.from_numpy(np.ascontiguousarray(np.transpose(features, (2, 0, 1)))).float()
        specular_in = torch.from_numpy(np.ascontiguousarray(np.transpose(specular_in, (2, 0, 1)))).float()
        specular_ref = torch.from_numpy(np.ascontiguousarray(np.transpose(specular_ref, (2, 0, 1)))).float()
        return {
        "seg":features, 
        "specular_in" : specular_in, 
        "specular_ref" : specular_ref,
        'NOISY_path': NOISY_path, 'GT_path': GT_path,
        "x_offset": 128,
        "y_offset": 128}

    # def __getitem__(self, index):
    #     GT_path, NOISY_path = None, None
    #     GT_size = self.opt['GT_size']

    #     GT_path = self.paths_GT[index]
    #     img_GT = util_disney.loadDisneyEXR_ref(GT_path)
       
    #     NOISY_path = self.paths_NOISY[index]
    #     img_NOISY = util_disney.loadDisneyEXR_feature(NOISY_path)

    #     # do the cropping
    #     H, W, _ = img_GT.shape
    #     x = random.randint(0, np.maximum(0, W - GT_size))
    #     y = random.randint(0, np.maximum(0, H - GT_size))
    #     img_GT = util._crop(img_GT, (y,x), GT_size)
    #     img_NOISY = util._crop(img_NOISY, (y,x), GT_size)

    #     if self.opt['phase'] == 'train':
    #         # augmentation - flip, rotate
    #         img_NOISY, img_GT = util.augment([img_NOISY, img_GT], self.opt['use_flip'], \
    #             self.opt['use_rot'])

    #     img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
    #     img_NOISY = torch.from_numpy(np.ascontiguousarray(np.transpose(img_NOISY, (2, 0, 1)))).float()
    #     return {'NOISY': img_NOISY, 'GT': img_GT,  'NOISY_path': NOISY_path, 'GT_path': GT_path}


    def __len__(self):
        return len(self.paths_GT)
