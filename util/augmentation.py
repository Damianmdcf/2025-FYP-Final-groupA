import cv2
import numpy as np
from skimage.util import random_noise
from skimage.morphology import disk
import random

def apply_clahe(img):
    """
    Apply CLAHE augmentation to image
    """

    # Convert to LAB and apply CLAHE only to lightness channel to add contrast in the picture
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab) # Split the channels

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l) # Applies CLAHE only to to lightness channel

    # Merge the processed lightness channel with the original a and b channels
    lab_clahe = cv2.merge((cl, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB) # Convert from LAB to RGB

    return img_clahe

def noise_augmentation(img):
    """
    Apply gaussian noise augmentation
    """
    # Add the random noise with a variance 0.1
    img_noisy = random_noise(img,var=0.1)
    # Turn the image back into an 8-bit image
    img_noisy_uint8 = (img_noisy * 255).astype(np.uint8)

    return img_noisy_uint8

def roughen_border(mask_input, max_perturb=5, n_points=200):
    """
    Apply a more jagged border to the mask of the image.
    n_point number of pixel will at random be found on the border of the lesion mask.
    Each of the pixels will then be moved at random a distance of max_perturb in any direction.
    Finally a new contour will be found, to compute a new mask with the newly randomly moved points,
    which results in a more jagged border of the mask.
    """
    # Turn the image into an 8-bit image
    mask = mask_input.astype(np.uint8)
    # Get the contour (i.e. border) of the image
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # fallback if no contour could be found
    if not contours:
        return mask_input  # Return original if no contour found

    # Shouldn't happen, but if multiple contours are found, we use only the biggest one
    contour = max(contours, key=cv2.contourArea) 
    
    # get the number of points on the contour
    num_contour_points = contour.shape[0]
    
    # fallback if the contour is empty
    if num_contour_points == 0:
        return mask_input # Should not happen if contours list is not empty, but good check

    # Adjust n_points if it's larger than the available contour points
    actual_n_points_to_sample = min(n_points, num_contour_points)

    # another fallback
    if actual_n_points_to_sample == 0 : # e.g. if n_points was 0 or num_contour_points was 0
        return mask_input # again, this shouldn't happen

    # Perturb some contour points randomly
    # Squeeze the contour after ensuring it's valid and we know num_contour_points
    contour_squeezed = contour.squeeze(axis=1) # Squeeze the middle dimension
    # If contour_squeezed is now 1D (e.g. single point (2,)), reshape it to (1,2)
    if contour_squeezed.ndim == 1 and num_contour_points == 1:
        contour_squeezed = contour_squeezed.reshape(1, 2)

    perturbed = contour_squeezed.copy() # Use the squeezed version
    
    # Sample indices from the available contour points
    indices = np.random.choice(num_contour_points, size=actual_n_points_to_sample, replace=False)

    # go trough each of the choosen points
    for idx in indices:
        # for the point get distance in x and y direction is should be moved
        dx = np.random.randint(-max_perturb, max_perturb + 1)
        dy = np.random.randint(-max_perturb, max_perturb + 1)
        # move the point
        perturbed[idx, 0] += dx 
        perturbed[idx, 1] += dy

    # Create new mask of all zeros
    new_mask = np.zeros_like(mask)

    # Reshape the changed array so it fits the expected shape of drawContours
    perturbed_reshaped = perturbed.reshape((-1, 1, 2)).astype(np.int32)

    # Create the new mask with the perturbed points.
    cv2.drawContours(new_mask, [perturbed_reshaped], -1, 1, thickness=cv2.FILLED)

    return new_mask