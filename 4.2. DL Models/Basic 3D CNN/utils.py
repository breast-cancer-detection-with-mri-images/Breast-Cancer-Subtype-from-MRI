import torch
import pandas

import numpy


def normalize_sitk_image(arr):
    '''Function to scale a simple-itk image array to values between 0 and 1
    
    Arguments:
    1. arr: a numpy array for a SimpleITK image
    
    Returns:
    1. scaled_arr: a scaled array with elements between 0 and 1.'''

    return (arr - arr.min())/(arr.max() - arr.min())