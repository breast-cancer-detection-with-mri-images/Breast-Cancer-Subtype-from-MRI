import torch
import pandas

import numpy
import os
import SimpleITK as sitk


def normalize_sitk_image(arr):
    '''Function to scale a simple-itk image array to values between 0 and 1
    
    Arguments:
    1. arr: a numpy array for a SimpleITK image
    
    Returns:
    1. scaled_arr: a scaled array with elements between 0 and 1.'''

    return (arr - arr.min())/(arr.max() - arr.min())


class MRI_Dataset_within_ROI(torch.utils.data.Dataset):
    '''Dataset for loading MRI sequences with only the tumour ROI enclosed.'''
    def __init__(self,
                 src_path,
                 dataframe,
                 seg_bb,
                 transform,
                 upscale,
                 augment = None,
                 sequence = 'post_1.img.gz'
                 ):
        '''Init method

        Arguments:
        1. src_path: the path to the processed NIFTI Dataset
        2. dataframe: the file consisting of patient-class characterisations
        3. seg_bb: segmentation bounding boxes data (dataframe)
        4. transform: the transformation excluding the upscaling for the 3D volume
        5. upscale: the upscaling transform to convert sequences to a standard format
        6. augment: augmentation for the 3D voxel tensor
        7. sequence: the sequence name (eg. 'post_1.img.gz')

        Returns:
        1. MRI_Dataset_within_ROI dataset
        '''
        self.src_path = src_path
        self.df = dataframe.to_numpy()
        self.seg_bb = seg_bb
        self.transform = transform
        self.sequence = sequence
        self.upscale = upscale
        self.augment = augment

    def __len__(self):
        '''Function to get the length of the dataset'''
        return len(self.df)

    def __getitem__(self, idx):
        '''Function to fetch item at an index of the dataset

        Arguments:
        1. idx: index

        Returns:
        1. 3D tensor for the tumour volume
        2. label
        '''
        patient, label = self.df[idx]
        label = torch.tensor(label)
        
        path = os.path.join(self.src_path, patient, self.sequence)
        img = sitk.ReadImage(path)
        arr = sitk.GetArrayFromImage(img)
        row1, row2, col1, col2, slice1, slice2 = self.seg_bb.loc[patient].tolist()

        segment = torch.tensor(arr[slice1: slice2, row1: row2, col1: col2].astype('float32'))[None, ...]
        segment = self.transform(segment).type(torch.float32)
    
        segment = self.upscale(segment)
        if self.augment is not None:
            segment = self.augment(segment)

        label = label.type(torch.LongTensor)
        
        return segment, label
    
    
class MRI_Dataset_within_ROI_both_prepost(torch.utils.data.Dataset):
    '''Dataset for loading MRI sequences with only the tumour ROI enclosed.'''
    def __init__(self,
                 src_path,
                 dataframe,
                 seg_bb,
                 transform,
                 upscale,
                 augment = None
                 ):
        '''Init method

        Arguments:
        1. src_path: the path to the processed NIFTI Dataset
        2. dataframe: the file consisting of patient-class characterisations
        3. seg_bb: segmentation bounding boxes data (dataframe)
        4. transform: the transformation excluding the upscaling for the 3D volume
        5. upscale: the upscaling transform to convert sequences to a standard format
        6. augment: augmentation for the 3D voxel tensor

        Returns:
        1. MRI_Dataset_within_ROI dataset
        '''
        self.src_path = src_path
        self.df = dataframe.to_numpy()
        self.seg_bb = seg_bb
        self.transform = transform
        self.upscale = upscale
        self.augment = augment

    def __len__(self):
        '''Function to get the length of the dataset'''
        return len(self.df)

    def __getitem__(self, idx):
        '''Function to fetch item at an index of the dataset

        Arguments:
        1. idx: index

        Returns:
        1. 3D tensor for the tumour volume
        2. label
        '''
        patient, label = self.df[idx]
        label = torch.tensor(label)
        
        pre_path = os.path.join(self.src_path, patient, 'pre.img.gz')
        post_path = os.path.join(self.src_path, patient, 'post_1.img.gz')

    
        row1, row2, col1, col2, slice1, slice2 = self.seg_bb.loc[patient].tolist()
        pre_img = self.get_arrs(pre_path, row1, row2, col1, col2, slice1, slice2)
        post_img = self.get_arrs(post_path, row1, row2, col1, col2, slice1, slice2)

        
        segment = torch.concat([pre_img, post_img])
        
        if self.augment is not None:
            segment = self.augment(segment)

        label = label.type(torch.LongTensor)
        
        return segment, label
    

    def get_arrs(self, path, row1, row2, col1, col2, slice1, slice2):
        img = sitk.GetArrayFromImage(sitk.ReadImage(path))[slice1: slice2, row1: row2, col1: col2]
        img = torch.Tensor(img.astype('float32'))[None, ...]
        return self.upscale(self.transform(img))
        
        
        