import os
import cv2
import numpy as np
from skimage import morphology

def border(mask):
    
    #Scans binary mask and returns outlines of every "white" region
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 

    #Calculagtes the area of every region and returns the biggest one, in this case our legion 
    lesion_contour= max(contours, key= cv2.contourArea) 

    lesion_area = cv2.contourArea(lesion_contour)

    # True tells OpenCV the contour is closed, so it adds the distance between the last and first point.
    border_perimeter = cv2.arcLength(lesion_contour, True) 

    if lesion_area == 0:
        irregularity = 0
    else:
        irregularity = (border_perimeter ** 2) / (4 * np.pi * lesion_area)
    
    # plt.imshow(mask, cmap="gray", interpolation="nearest")
    # plt.axis("off")
    # plt.show()

    return irregularity

# Philip: This is an alternative to the function above. They both output compactness, but using two different approaches.
# I have compared both and their outputs are similar except that measure_streaks() sometimes wrongly outputs 0, so there
# maybe is some bug the code.
def get_compactness(mask):
    # Calculate area by summing over the white pixels in the binarized mask
    area = np.sum(mask)

    # Get the structuring element, which in this case is a disk
    struct_el = morphology.disk(3)
    
    # Use the disk to erode (remove) only the pixels at the border
    mask_eroded = morphology.binary_erosion(mask, struct_el)

    # Get the perimeter by counting the number of pixel that was eroded above
    perimeter = np.sum(mask ^ mask_eroded)

    # Return the compactness using the formla seen below
    return perimeter**2 / (4 * np.pi * area)