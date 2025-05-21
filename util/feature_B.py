import os
import cv2
import numpy as np
import matplotlib as plt

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
