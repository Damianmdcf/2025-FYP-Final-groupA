import cv2
import numpy as np
from skimage.util import random_noise
import random

def apply_clahe(img):

    # Convert to LAB and apply CLAHE only to lightness channel to add contrast in the picture
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab) # SPlit the channels

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l) # Applies CLAHE only to to lightness channel

    # Merge the processed lightness channel with the original a and b channels
    lab_clahe = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB) # Convert from LAB to RGB

    return img_clahe

def apply_flip(img, mask=None):

    code = random.randint(-1, 1) # -1: flip vertically, 0: flip horizontally, 1: flip both

    flipped_img = cv2.flip(img, code)
    flipped_mask = cv2.flip(mask, code) if mask is not None else None
    
    return flipped_img, flipped_mask

def noise_augmentation(img):
    img_noisy = random_noise(img,var=0.1)
    img_noisy_uint8 = (img_noisy * 255).astype(np.uint8)
    return img_noisy_uint8
