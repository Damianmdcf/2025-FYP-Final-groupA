import cv2
import numpy as np
from skimage.util import random_noise
from skimage.morphology import disk
import random

def apply_clahe(img):

    # Convert to LAB and apply CLAHE only to lightness channel to add contrast in the picture
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
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
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # print(f"Debug roughen_border: No contours found for a mask with sum {np.sum(mask_input)}")
        return mask_input  # Return original if no contour found

    # Consider using the largest contour if multiple are found, though RETR_EXTERNAL often gives one main one.
    contour = max(contours, key=cv2.contourArea) 
    
    # Squeeze might be problematic if contour has only 1 point.
    # Check number of points in the contour.
    # contour is typically of shape (num_points, 1, 2).
    num_contour_points = contour.shape[0]

    if num_contour_points == 0:
        # print("Debug roughen_border: Contour found but has 0 points.")
        return mask_input # Should not happen if contours list is not empty, but good check

    # Adjust n_points if it's larger than the available contour points
    actual_n_points_to_sample = min(n_points, num_contour_points)

    if actual_n_points_to_sample == 0 : # e.g. if n_points was 0 or num_contour_points was 0
        # print("Debug roughen_border: No points to sample for perturbation.")
        return mask_input


    # Perturb some contour points randomly
    # Squeeze the contour after ensuring it's valid and we know num_contour_points
    contour_squeezed = contour.squeeze(axis=1) # Squeeze the middle dimension
    # If contour_squeezed is now 1D (e.g. single point (2,)), reshape it to (1,2)
    if contour_squeezed.ndim == 1 and num_contour_points == 1:
        contour_squeezed = contour_squeezed.reshape(1, 2)


    perturbed = contour_squeezed.copy() # Use the squeezed version
    
    # Sample indices from the available contour points
    # Ensure replace=False is only used if actual_n_points_to_sample <= num_contour_points (which it is by min())
    indices = np.random.choice(num_contour_points, size=actual_n_points_to_sample, replace=False)

    for idx in indices:
        dx = np.random.randint(-max_perturb, max_perturb + 1)
        dy = np.random.randint(-max_perturb, max_perturb + 1)
        # Ensure perturbed[idx] is treated as a 2-element array for addition
        perturbed[idx, 0] += dx 
        perturbed[idx, 1] += dy

    # Create new mask
    new_mask = np.zeros_like(mask)
    # Reshape perturbed for drawContours: needs to be (N, 1, 2)
    perturbed_reshaped = perturbed.reshape((-1, 1, 2)).astype(np.int32)
    cv2.drawContours(new_mask, [perturbed_reshaped], -1, 1, thickness=cv2.FILLED) # Use cv2.FILLED for thickness=-1

    return new_mask