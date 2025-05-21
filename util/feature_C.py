import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage.color import rgb2hsv
from scipy.stats import circmean, circvar
from statistics import variance


# Function to measure color irregularity

def get_irregularity_score(image_rgb, mask):
    # Change image from BGR to RGB, update: already included in DataLoader
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get the segmented picture
    slic_segments = slic_segmentation(image_rgb, mask)

    # Value was not included, as its strongly related to the light in which the photo was taken
    hue_var, sat_var = hsv_variance(image_rgb, slic_segments)

    # Combined mean of the hue and saturation variance
    irregularity_score = 0.5 * hue_var + 0.5 * sat_var

    return irregularity_score



# Divides an image into superpixels — groups of connected pixels that share similar color and texture
def slic_segmentation(image, mask, n_segments = 200, compactness = 10): 
    # N_segments is approximate number of labels/regions, 100 is a default, so we choose 200 as it is a small "uniform" lession
    # Compactnes 10 is default
    slic_segments = slic(image,
                    n_segments = n_segments,
                    compactness = compactness,
                    sigma = 1,
                    mask = mask,
                    start_label = 1,
                    channel_axis = 2)
    
    return slic_segments


def get_hsv_means(image, slic_segments):
    # Convert images from RGB to HSV, as it better explains color irregularity
    hsv_image = rgb2hsv(image) 

    # Get the actual amount of segments created (not nescesarily the default amount)
    max_segment_id = np.unique(slic_segments)[-1]

    # Compute and collect all HSV means across all segments
    hue_mean = []
    sat_mean = []
    val_mean = []

    for i in range(1, max_segment_id + 1):

        segment = hsv_image.copy()
        segment[slic_segments != i] = np.nan

        hue_mean.append(circmean(segment[:, :, 0], high=1, low=0, nan_policy='omit')) # The mean of hue uses circmean, since hue is on a circular scale (color wheel)
        sat_mean.append(np.mean(segment[:, :, 1], where = (slic_segments == i)))
        val_mean.append(np.mean(segment[:, :, 2], where = (slic_segments == i))) 
        
    return hue_mean, sat_mean, val_mean


def hsv_variance(image, slic_segments): #based on hsv variance
    
    #Checking if we have enough segments to calculate any variance
    if len(np.unique(slic_segments)) == 2: 
        return 0, 0, 0

    hue_mean, sat_mean, val_mean = get_hsv_means(image, slic_segments)
     
    n = len(hue_mean)

    # Compute the variance of hue and saturations means across all segments, we omit value since it is unused
    hue_var = circvar(hue_mean, high=1, low=0)
    sat_var = variance(sat_mean, sum(sat_mean)/n)
    #val_var = variance(val_mean, sum(val_mean)/n) 

    return hue_var, sat_var

mask2= cv2.imread("PAT_82_125_365_mask.png", cv2.IMREAD_GRAYSCALE)
image2= cv2.imread("PAT_82_125_365.png")
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

segment2= slic_segmentation(image2, mask2)
print(get_irregularity_score(image2, segment2))





