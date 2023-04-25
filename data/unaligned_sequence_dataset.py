import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import torch

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from common.imutils import imshowpts
from common.fileutils import get_dirs_list, adjust_idx_if_out_of_bounds
import json
import yaml
from random import randint

from data.datautils import resize_img,prepare_images,makeseq, concat_rgb_with_iuv, load_packets_utils
from data.datautils import get_centroids_path,load_centroids_images

MISSING_VALUE = -1
class UnalignedSequenceDataset(BaseDataset):
    def initialize(self, opt,data_cfgs):
        self.opt = opt
        self.use_fullseq = True if not hasattr(opt, 'use_fullseq') else opt.use_fullseq

        obj1 = data_cfgs['DATASET']['OBJ1']
        obj2 = data_cfgs['DATASET']['OBJ2']

        self.AP_RGB_paths, self.A_RGB_objid = makeseq(obj1['RGB_DATA_ROOT'], filters=obj1['USED_DIRS'])
        self.AP_IUV_paths, self.A_IUV_objid = makeseq(obj1['IUV_DATA_ROOT'], filters=obj1['USED_DIRS'])
        self.AP_paths = concat_rgb_with_iuv(self.AP_RGB_paths, self.AP_IUV_paths)
        
        self.BP_RGB_paths, self.B_RGB_objid = makeseq(obj2['RGB_DATA_ROOT'], filters=obj2['USED_DIRS'])
        self.BP_IUV_paths, self.B_IUV_objid = makeseq(obj2['IUV_DATA_ROOT'], filters=obj2['USED_DIRS'])
        self.BP_paths = concat_rgb_with_iuv(self.BP_RGB_paths, self.BP_IUV_paths)

        clistA_RGB = [os.path.join(obj1['CENTROIDS']['RGB_DATA_ROOT'], item) for item in obj1['CENTROIDS']['IMAGES']]
        ClistA_IUV = [os.path.join(obj1['CENTROIDS']['IUV_DATA_ROOT'], item) for item in obj1['CENTROIDS']['IMAGES']]
        clistA = concat_rgb_with_iuv([clistA_RGB], [ClistA_IUV])[0]
        self.centroidsA = load_centroids_images(clistA , iuv_mode='iuv1')
        
        clistB_RGB = [os.path.join(obj2['CENTROIDS']['RGB_DATA_ROOT'], item) for item in obj2['CENTROIDS']['IMAGES']]
        ClistB_IUV = [os.path.join(obj2['CENTROIDS']['IUV_DATA_ROOT'], item) for item in obj2['CENTROIDS']['IMAGES']]
        clistB = concat_rgb_with_iuv([clistB_RGB], [ClistB_IUV])[0]
        self.centroidsB = load_centroids_images(clistB, iuv_mode='iuv1')

        self.A_size = sum([len(i) for i in self.AP_paths])
        self.B_size = sum([len(i) for i in self.BP_paths])

        rgba_transform_list = [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5, 0.5),(0.5, 0.5, 0.5, 0.5))] 
        self.rgba_transform = transforms.Compose(rgba_transform_list)
        
        rgb_transform_list = [transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))] 
        self.rgb_transform = transforms.Compose(rgb_transform_list)
        
        self.subseq_idx = 0
        self.packet_len = 3 if not hasattr(opt,'seq_len') else opt.seq_len

        if not self.use_fullseq :
            self.subseq_len = 30
            self.start_new_subseq()
        else :
            self.currA_seq = 0
            self.currB_seq = 0
            self.num_seq = len(self.A_paths)
            self.start_fullseq()


    def choose_subseq(self, paths, ):
        
        if len(paths) == 1:
            curr_seq = 0
        else:
            curr_seq = randint(0, len(paths) - 1)
        if self.subseq_len > len(paths[curr_seq]) - 1 and len(paths[curr_seq]) > 30:
            self.subseq_len = 30
        else:
            self.subseq_len = 10
        curr_idx = randint(0, len(paths[curr_seq]) - self.subseq_len)
        return curr_seq,curr_idx

    def start_fullseq(self):
        self.subseq_idx = 0

        self.currA_idx = 0
        self.currB_idx = 0

        self.currB_keys = self.centroidsB
        self.currA_keys = self.centroidsA

        self.subseq_len = len(self.AP_paths[self.currA_seq])

    def start_new_subseq(self):
        self.subseq_idx = 0

        self.currA_seq, self.currA_idx = self.choose_subseq(self.AP_paths)
        self.currB_seq, self.currB_idx = self.choose_subseq(self.BP_paths)
        self.currB_keys = self.centroidsB
        self.currA_keys = self.centroidsA

    def get_current_keys(self):
        return self.currB_keys, self.currA_keys

    def load_packets(self,itemlist,ityep = 'img'):
        imglist = []
        for f in itemlist:
            if isinstance(f, tuple) == False:
                AP_img = Image.open(f).convert('RGBA')  # SM - changed to RGBA (instead of RGB)
                AP0 = load_packets_utils(AP_img, 'img', self.opt.loadSize, self.rgba_transform, self.opt.fineSize)
            else:
                rgb_image = Image.open(f[0]).convert('RGBA')
                rgb,offset = load_packets_utils(rgb_image, 'img', self.opt.loadSize, self.rgba_transform, self.opt.fineSize)
                iuv_image = Image.open(f[1]).convert('RGB')
                iuv, _ = load_packets_utils(iuv_image, 'iuv1', self.opt.loadSize, self.rgb_transform, self.opt.fineSize, offset)
                AP0 = torch.cat((rgb, iuv), 1)
            imglist.append(AP0)
        packet = np.concatenate(imglist, axis=0)
        return packet
    

    def __getitem__(self, index):
        if self.subseq_idx == self.subseq_len :
            if not self.use_fullseq :
                self.start_new_subseq()
            else :
                self.currA_seq = (self.currA_seq + 1) % len(self.AP_paths)
                self.currB_seq = (self.currB_seq + 1) % len(self.BP_paths)
                self.start_fullseq()
                
        A_idx = self.currA_idx + self.subseq_idx % (len(self.AP_paths[self.currA_seq]) - self.packet_len)
        A_idx = adjust_idx_if_out_of_bounds(A_idx, len(self.AP_paths[self.currA_seq]), self.packet_len)
        
        B_idx = self.currB_idx + self.subseq_idx % (len(self.BP_paths[self.currB_seq]) - self.packet_len)
        B_idx = adjust_idx_if_out_of_bounds(B_idx, len(self.BP_paths[self.currB_seq]), self.packet_len)

        AP_path = self.AP_paths[self.currA_seq][A_idx : A_idx + self.packet_len]
        BP_path = self.BP_paths[self.currB_seq][B_idx : B_idx + self.packet_len]
        

        AP0 = self.load_packets(AP_path, ityep='img')
        BP0 = self.load_packets(BP_path, ityep='img')

        if self.subseq_idx == 0 :
            B_keys = self.currB_keys
            A_keys = self.currA_keys
        else :
            B_keys, A_keys = [], []

        self.subseq_idx += 1

        
        return {'AP0': AP0, 'BP0': BP0, 'imgidx' : self.subseq_idx ,
                'A_keys' : A_keys, 'B_keys' : B_keys}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedSequenceDataset'


if __name__ == '__main__':
    pass