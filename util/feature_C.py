import numpy as np
from skimage.segmentation import slic
from skimage.color import rgb2hsv
from scipy.stats import circmean, circvar
from statistics import variance

def get_irregularity_score(image_rgb, mask):
    """
    Main function that gets the color irregularity used as the C feature in the ABC model.
    Returned value is variance of the means of each superpixel on both the hue and saturation color channels.
    """
    # Get the segmented picture
    slic_segments = slic_segmentation(image_rgb, mask)
    # Get the variance of the hue and the saturation across each SLIC segment.
    # Value was not included, as its strongly related to the light in which the photo was taken
    hue_var, sat_var = hsv_variance(image_rgb, slic_segments)
    # Combined mean of the hue and saturation variance where we weight the importance of each of them equally
    irregularity_score = 0.5 * hue_var + 0.5 * sat_var

    return irregularity_score

def slic_segmentation(image, mask, n_segments = 200, compactness = 10): 
    """
    Helper function returning the number of superpixels.
    Superpixels is groups of connected pixels that share similar color and texture and are located close together.
    N_segments is approximate number of labels/regions, 100 is a default, so we choose 200 as it is a small "uniform" lesion.
    Compactnes 10 is the default.
    """
    # Compute the slic segments
    slic_segments = slic(image,
                    n_segments = n_segments,
                    compactness = compactness,
                    sigma = 1,
                    mask = mask,
                    start_label = 1,
                    channel_axis = 2)
    return slic_segments # Returns the number of segments


def get_hsv_means(image, slic_segments):
    """
    Get the HSV means for each of the segments
    """
    # Convert images from RGB to HSV, as it better explains color irregularity
    hsv_image = rgb2hsv(image) 
    # Get the actual amount of segments created (not nescesarily the default amount)
    max_segment_id = np.unique(slic_segments)[-1]

    # Compute and collect all HSV means across all segments
    hue_mean = []
    sat_mean = []
    val_mean = []
    for i in range(1, max_segment_id + 1): # looping through each of the superpixels
         # copy the original image
        segment = hsv_image.copy()
        # mask out all other segments such that only HSV values for segment i is left
        segment[slic_segments != i] = np.nan
        # Slice and get hue mean of the segment. The mean of hue uses circmean, since hue is on a circular scale (color wheel)
        hue_mean.append(circmean(segment[:, :, 0], high=1, low=0, nan_policy='omit'))
        # Slice and get saturation mean of the segment.
        sat_mean.append(np.mean(segment[:, :, 1], where = (slic_segments == i)))
        # Slice and get value mean of the segment.
        val_mean.append(np.mean(segment[:, :, 2], where = (slic_segments == i))) 
        
    return hue_mean, sat_mean, val_mean # return the 3 means


def hsv_variance(image, slic_segments):
    """
    Gets the variance of hue and saturations means across all segments
    """
    # Checking if we have enough segments to calculate any variance
    if len(np.unique(slic_segments)) == 2:
        return 0, 0, 0 # Fallback
    # Calls the function above to get the 3 means
    hue_mean, sat_mean, val_mean = get_hsv_means(image, slic_segments)
    n = len(hue_mean)
    # Compute the variance of hue and saturations means across all segments, we omit value since it is unused
    hue_var = circvar(hue_mean, high=1, low=0)
    sat_var = variance(sat_mean, sum(sat_mean)/n)

    return hue_var, sat_var # return the variance of the 2 channels of interest





