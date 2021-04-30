# -*- coding: utf-8

import os

import pandas as pd
from PIL import Image
import nibabel as nib
import numpy as np
import torch
from torch.utils import data

class MinMaxNormalize(object):
    "Normalizing the ndarray between zero and one using its minimum and maximum value"
    def __call__(self,sample):
        x = sample
        xmin = x.min()
        xmax = x.max()
        x = x-xmin
        if (xmax-xmin) != 0:
            x = x/(xmax-xmin)
        
        return x

class CenterCrop(object):
    """Crops the given np.ndarray at the center to have a region of
    """
    def __call__(self, sample):
        w, h = sample.shape[1], sample.shape[0]
        th, tw = self.finesize
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return sample[y1:y1+th, x1:x1+tw, :]

class convert(object):
    "convert to Tensor from numpy"
    def __call__(self,sample):
         x = sample
#         x = x.transpose(1,0)
         x = x.transpose(2,0,1).astype(np.float32)#converting from HXWXC to CXHXW
         return torch.from_numpy(x)
     

class ScanDataset(data.Dataset):
    """Scan dataset

    Args:
        csv_file: a 11-column csv_file, column one contains the names of image files, column 2-11 contains the empiricial distributions of ratings
        root_dir: directory to the images
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        #self.annotations = pd.read_csv(csv_file, index_col=0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
#        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]) + '.png')
        img_name = os.path.join(self.root_dir, str(self.annotations.iloc[idx, 0]) + '.nii.gz')
#        image = Image.open(img_name)
        image_obj = nib.load(img_name)
        img = image_obj.get_fdata()
        image =  img[:,:,6]
        
        xmin = image.min()
        xmax = image.max()
        image = image-xmin
        if (xmax-xmin) != 0:
            image = image/(xmax-xmin)       
            
        image = np.stack((image,)*3,axis=-1)
        image =  Image.fromarray(np.uint8(image*256), 'RGB')    
        
        annotations = np.array(self.annotations.iloc[idx, 1])
        
        if annotations == True:
            annotations = np.append(annotations,[0.0])
        else:
            annotations = np.append(annotations,[1.0])
        annotations = annotations.astype('float').reshape(-1, 1)

        sample = {'img_id': img_name, 'image': image, 'annotations': annotations}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
