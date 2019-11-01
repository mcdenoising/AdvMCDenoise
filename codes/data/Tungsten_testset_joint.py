
import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util
import data.util_disney as util_disney
import data.util_exr as exr_utils
import OpenEXR
# for Disney dataset
def get_distinct_prefix(dir_path):
    names = set()
    for f in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, f)):
            names.add(f.split(".")[0].rsplit("_",1)[0])
    return list(names)

def load_reference_mat(path_ref, scene_name, path_feature=None):
    color_path = os.path.join(path_ref, scene_name)
    inFile = exr_utils.open(color_path)
    color = inFile.get_all()["default"]    
    color = np.log(color+1)
    return color

class TungstenTestset(data.Dataset):
    def __init__(self, opt):
        super(TungstenTestset, self).__init__()
        self.opt = opt
        self.paths_GT = []
        self.paths_NOISY = []

        feature_dir = opt['dataroot_NOISY']
        distinct_prefixs = get_distinct_prefix(feature_dir) # scene_spp
        SCENE_LIST = [ "bathroom","bathroom2", "car", "living-room", "living-room-2", "living-room-3"]#"car2",
        for distinct_prefix in distinct_prefixs:
            scece_name = distinct_prefix.split("_")[0]
            spp = distinct_prefix.split("_")[1]
            # if scece_name not in SCENE_LIST or spp not in ["32"]:
            #     continue
            # self.paths_GT.append(os.path.join(opt["dataroot_GT"], scece_name + ".exr") )
            self.paths_NOISY.append(os.path.join(opt["dataroot_NOISY"], distinct_prefix + ".exr") )
        # self.paths_GT = sorted(self.paths_GT)    
        self.paths_NOISY = sorted(self.paths_NOISY)
        
    def __getitem__(self, index):
        # print("[INFO]getting item from dataloader.... index = %d len(paths_NOISY)=%d" % (index,len(self.paths_NOISY)))
        GT_path, NOISY_path = None, None
        # GT_size = self.opt['GT_size']
        NOISY_path = self.paths_NOISY[index]
        scece_name = os.path.basename(NOISY_path).split(".")[0].split("_")[0]

        GT_path = os.path.join(self.opt["dataroot_GT"], scece_name + ".exr")
      
        # img_GT = load_reference_mat("",GT_path)
        img_GT, diffuse_ref, specular_ref, features_ref = util.load_feature_mat_complete_tungsten_joint(GT_path, self.opt["feature"]+["diffuse"]) 
        img_NOISY, diffuse, specular, features = util.load_feature_mat_complete_tungsten_joint(NOISY_path,  self.opt["feature"]+["diffuse"]) 
        
        if(diffuse_ref is None):
            img_GT = load_reference_mat("",GT_path)
            diffuse_ref = diffuse.copy()
            specular_ref = specular.copy()
        

        y = img_GT.shape[1] #(1280 - img_GT.shape[1])//2 #800
        x = img_GT.shape[0] #(1280 - img_GT.shape[0])//2

        # img_GT = np.pad(img_GT,  [(x, x), (y, y), (0, 0)], mode="constant", constant_values=(0.0) )
        # diffuse_ref = np.pad(diffuse_ref,  [(x, x), (y, y), (0, 0)], mode="constant", constant_values=(0.0) )
        # specular_ref = np.pad(specular_ref,  [(x, x), (y, y), (0, 0)], mode="constant", constant_values=(0.0) )

        
       
        # img_NOISY = np.pad(color,  [(x, x), (y, y), (0, 0)], mode="constant", constant_values=(0.0) )
        # diffuse = np.pad(diffuse,  [(x, x), (y, y), (0, 0)], mode="constant", constant_values=(0.0) )
        # specular = np.pad(specular,  [(x, x), (y, y), (0, 0)], mode="constant", constant_values=(0.0) )
        # # features = np.pad(features,  [(x, x), (y, y), (0, 0)], mode="constant", constant_values=(0.0) )
        # # color, diffuse, specular, features = util.load_feature_mat_complete_tungsten_joint(NOISY_path) 
        # # img_NOISY = np.pad(color,  [(x, x), (y, y), (0, 0)], mode="edge" )
        # # diffuse = np.pad(diffuse,  [(x, x), (y, y), (0, 0)], mode="edge" )
        # # specular = np.pad(specular,  [(x, x), (y, y), (0, 0)], mode="edge" )
        # features = np.pad(features,  [(x, x), (y, y), (0, 0)], mode="edge" )
        # # print("img_NOISY.shape after padding", img_NOISY.shape)


        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        specular_ref = torch.from_numpy(np.ascontiguousarray(np.transpose(specular_ref, (2, 0, 1)))).float()
        diffuse_ref = torch.from_numpy(np.ascontiguousarray(np.transpose(diffuse_ref, (2, 0, 1)))).float()

        img_NOISY = torch.from_numpy(np.ascontiguousarray(np.transpose(img_NOISY, (2, 0, 1)))).float()
        diffuse_in = torch.from_numpy(np.ascontiguousarray(np.transpose(diffuse, (2, 0, 1)))).float()
        specular_in = torch.from_numpy(np.ascontiguousarray(np.transpose(specular, (2, 0, 1)))).float()
        features = torch.from_numpy(np.ascontiguousarray(np.transpose(features, (2, 0, 1)))).float()

        if NOISY_path is None:
            NOISY_path = GT_path


        return {'NOISY': img_NOISY, 
        "seg":features, 
        "diffuse_in" : diffuse_in, 
        "specular_in" : specular_in, 
        'GT': img_GT,
        "diffuse_ref" : diffuse_ref,
        "specular_ref" : specular_ref,
        'NOISY_path': NOISY_path, 
        'GT_path': GT_path,
        "x_offset": x,
        "y_offset": y
        }


    def __len__(self):
        return len(self.paths_NOISY)
