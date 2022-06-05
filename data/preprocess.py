import torch
import numpy as np

def img_to_tensor(cvImage):
    '''
    :param cvImage: numpy HWC BGR 0~255
    :return: tensor img CHW BGR  float32 cpu 0~1
    '''

    cvImage = cvImage.transpose(2, 0, 1).astype(np.float32) / 255


    return torch.from_numpy(cvImage)

def mask_to_tensor(cvImage):
    '''
    :param cvImage: numpy 0(background) or 255(crack pixel)
    :return: tensor 0 or 1 float32
    '''
    cvImage = cvImage.astype(np.float32) / 255

    return torch.from_numpy(cvImage)