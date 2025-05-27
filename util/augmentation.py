import cv2
import numpy as np
from skimage.util import random_noise
from skimage.morphology import disk
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

def roughen_border(mask_input, max_perturb=5, n_points=200):
    mask = mask_input.astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask_input  # Return original if no contour found

    contour = contours[0].squeeze()

    # Perturb some contour points randomly
    perturbed = contour.copy()
    indices = np.random.choice(len(contour), size=n_points, replace=False)

    for idx in indices:
        dx = np.random.randint(-max_perturb, max_perturb + 1)
        dy = np.random.randint(-max_perturb, max_perturb + 1)
        perturbed[idx] = perturbed[idx][0] + dx, perturbed[idx][1] + dy

    # Create new mask
    new_mask = np.zeros_like(mask)
    perturbed = perturbed.reshape((-1, 1, 2)).astype(np.int32)
    cv2.drawContours(new_mask, [perturbed], -1, 1, thickness=-1)

    return new_mask