import cv2
import numpy as np

def getborder(mask):
    """
    Gets the compactness score used as the B feature in the ABC model
    """
    # Turn the mask into an 8-bit integer
    mask = mask.astype(np.uint8)
    # Scans binary mask and returns contour i.e. the border, mulitple if more than one lesion is found
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    # Get the biggest countour, so smaller "lesions" in the same image is ignored
    lesion_contour= max(contours, key= cv2.contourArea)
    # Get the area of the outlined lesion
    lesion_area = cv2.contourArea(lesion_contour)
    # Get the length of the perimeter of the lesion
    border_perimeter = cv2.arcLength(lesion_contour, True) 
    # Fallback, if no lesion could be found return 0
    if lesion_area == 0:
        irregularity = 0
    else:
        # Use the compactness formula to compute the border irregularity
        irregularity = (border_perimeter ** 2) / (4 * np.pi * lesion_area)
    return irregularity # return compactness where 1 is perfect circle and higher values means more irregular border