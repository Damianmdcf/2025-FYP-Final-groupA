import cv2
import numpy as np
import random

def flip_sample(img, mask=None):

    code = random.randint(-1, 1) # -1: flip vertically, 0: flip horizontally, 1: flip both

    flipped_img = cv2.flip(img, code)
    flipped_mask = cv2.flip(mask, code) if mask is not None else None
    
    return flipped_img, flipped_mask

