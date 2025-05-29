import numpy as np
from skimage.transform import rotate
import cv2


def get_asymmetry(mask):
    """
    Main function to get the asymmetry score used as the A feature in the ABC model
    """
    scores = []
    # Crop and rotate 6 times to cover 180 degrees to check asymmetry from multiple angles
    for _ in range(6):
        # crop the mask
        segment = crop(mask)
        # Add the asymmetry score for each rotation (the # pixels not overlapping divided by total white pixels)
        flipped = np.flip(segment)
        # fallback if the crop didn't include any of the segment
        if np.sum(segment) == 0:
            continue
        # compute the XOR to find the not overlapping
        score = np.sum(np.logical_xor(segment, flipped)) / (np.sum(segment))
        # append the score for this one segment
        scores.append(score)
        # Rotate 30 degrees before next iteration 
        mask = rotate(mask, 30)
        
    # Return the average mean score
    return sum(scores) / len(scores) if len(scores) > 0 else None 
    # range goes from 0 to 1 (theoretically) where 0 is perfect circle. In our case score can go above 1 if multiple leasions are in the mask.


def crop(mask): 
        """
        Helper function to crop the lesion
        """
        # Finds the horizontal midpoint of the lesion
        mid = find_midpoint(mask)
        # Finds the coordinates of all white lesion pixels
        y_nonzero, x_nonzero = np.nonzero(mask)
        # Define the vertical bounds of the lesion
        y_lims = [np.min(y_nonzero), np.max(y_nonzero)]
        x_lims = np.array([np.min(x_nonzero), np.max(x_nonzero)]) # np.array since we need it in calculations
        # Finds the max distance from one of the x_lims to mid
        # This is how far we can crop without affecting the lesion
        x_dist = max(np.abs(x_lims - mid))
        # Makes new symmetric x limits (horisontal bounds)
        x_lims = [mid - x_dist, mid + x_dist]
        # return the cutout of the image, with the lesion centered
        return mask[y_lims[0]:y_lims[1], x_lims[0]:x_lims[1]]

def find_midpoint(mask):
        """
        Find the midpoint
        """
        # Counts the white pixels in each column
        summed = np.sum(mask, axis=0) 
        # Sums the white column pixels and takes half
        half_sum = np.sum(summed) / 2
        # Go through columns until the accumulated sum of prev. columns is more than half_sum
        for i, n in enumerate(np.add.accumulate(summed)): 
            if n > half_sum:
                return i # return the column where the middle of the lesion is