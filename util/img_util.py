import os
import cv2
import numpy as np
import pandas as pd

# import dotenv to load .env paths
from dotenv import load_dotenv
load_dotenv()

# get the path to where on your local machine images and masks lie.
# Remeber to create .env file as explain in README file.
images_path = os.getenv("IMAGE_DATA_URL_LOCAL")
mask_path = os.getenv("MASK_DATA_URL_LOCAL")

# import all utils
from .inpaint_util import removeHair
from .mask import get_mask
from .feature_A import get_asymmetry
from .feature_B import getborder
from .feature_C import get_irregularity_score
from .augmentation import apply_clahe, roughen_border, noise_augmentation

def readImageFile(file_path):
    """
    helper function to read the files
    """
    # read image as an 8-bit array
    img_bgr = cv2.imread(file_path)

    # convert to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # convert the original image to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    return img_rgb, img_gray

def get_binary_mask(mask_path):
    """
    helper function to turn the annotated mask into a binary array
    """
    # if an annotated mask is available
    if mask_path:
         # read image as an 8-bit array
        mask_img_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        # based on a threshold of 120, set black pixels to 0 and white pixels to 255
        _, img_th1 = cv2.threshold(mask_img_gray, 120, 255, cv2.THRESH_BINARY)
        # return the binary image
        return img_th1
    else:
        # if no annotated mask exists then just return none
        return None

def find_mask(image_path, mask_list, masks_filepath):
    """
    helper function to get the path to the annotated mask of the image of interest
    """
    # get the annotated mask file name of structure img_id_mask.png
    basename = os.path.basename(image_path).replace(".png", "_mask.png")
    # add the path to the folder with all mask images
    mask_path = masks_filepath + "/" + basename
    # if a mask exists then return the path to it
    if mask_path in mask_list:
        return mask_path
    else:
        # if not then return none
        return None

class ImageDataLoader:
    """
    Data loader used to load all images from local machine.
    Iterator makes it possible to perform hair removal, extract features and apply data augmentation
    in a for loop.
    """
    def __init__(self, directory, masks, hairless=None, augmentation=None):
        self.directory = directory # path to images folder on your computer
        self.masks = masks # path to annotated masks folder on your computer
        self.hairless = hairless # true if hair removal should be applied
        self.augmentation = augmentation # true if data augmentation should be applied

        # extract all images from images folder and sort them 
        self.file_list = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', 'jpeg', '.bmp', '.tiff'))])
        # extract all annotated masks from masks folder and sort them 
        self.mask_list = sorted([os.path.join(masks, f) for f in os.listdir(masks) if f.lower().endswith(('.png', '.jpg', 'jpeg', '.bmp', '.tiff'))])
        
        # check if images exists in images folder
        if not self.file_list:
            raise ValueError("No image files found in the directory.")
        
        # number of images
        self.num_batches = len(self.file_list)

        # load metadata
        self.df = pd.read_csv("data/metadata.csv")
        self.test_data = pd.read_csv('misc_data/test-baseline-data.csv')
        self.test_mel = list(self.test_data[self.test_data['Melanoma'] == 1]["img_id"])

    def is_melanoma(self, image_id):
        """
        Take img_id as input and outputs True if img_id has been diagnosed as melanoma
        """
        # get the diagnostic from metadata
        diagnostic = self.df.loc[self.df['img_id'] == image_id, "diagnostic"].iloc[0]
        
        # check that image is MEL and is not in test data to avoid data leakage
        if diagnostic == "MEL" and image_id not in self.test_mel:
            return True
        else:
            return False

    def __len__(self):
        """
        number of images
        """
        return self.num_batches

    def __iter__(self):
        """
        The iterator where all inpainting, feature extraction and augmentation happens.
        """
        # loop trough every image in the file list
        for filename in self.file_list:
            try: 
                # get the path to the annotated mask if it exists
                mask_path = find_mask(filename, self.mask_list, self.masks)
                # load the image
                img_rgb, img_gray = readImageFile(filename)
                # check if data augmentation should be applied
                if self.augmentation:
                    # get the image id from the filename
                    img_id = os.path.basename(filename)
                    # check if the image is MEL
                    mel = self.is_melanoma(img_id)

                    if mel:
                        # get a binary array of the annotated mask
                        manual_mask = get_binary_mask(mask_path)
                        # use the original image and the annotated mask to compute our own region growing mask
                        mask = get_mask(img_rgb, manual_mask)

                        # NOISE AUGMENTATION:
                        # Apply gaussian noise to the image
                        noise_img = noise_augmentation(img_rgb)
                        # Mask does not change since a region growing mask is not possible with noise in the image
                        noise_mask = mask

                        # CLAHE AUGMENTATION:
                        # Apply CLAHE to the image
                        contrast_img = apply_clahe(img_rgb)
                        # Create a new region growing mask with CLAHE image
                        contrast_mask = get_mask(contrast_img, manual_mask)
                        # fallback if the mask could not be compute and therefore it's all black
                        if np.sum(contrast_mask) == 0:
                            contrast_mask = mask

                        # EXTRA BORDER AUGMENTATION:
                        # Apply extra border to the mask of the original image
                        extra_border_mask = roughen_border(mask)
                        # fallback if the mask could not be compute and therefore it's all black
                        if np.sum(extra_border_mask) == 0:
                            extra_border_mask = mask 

                        # NOISE FEATURES COMPUTED:
                        assymetry_noise = get_asymmetry(noise_mask) # get assymetry for noise image (same as original)
                        _border_noise = getborder(noise_mask) # get border for noise image (same as original)
                        color_noise = get_irregularity_score(noise_img, noise_mask) # get color for noise image
                        # ASYMMETRY FEATURES COMPUTED:
                        assymetry_contrast = get_asymmetry(contrast_mask) # get assymetry for CLAHE image
                        _border_contrast = getborder(contrast_mask) # get border for CLAHE image
                        color_contrast = get_irregularity_score(contrast_img, contrast_mask) # get color for CLAHE image
                        # EXTAR BORDER FEATURES COMPUTED:
                        assymetry_extra_border = get_asymmetry(extra_border_mask.astype(bool)) # get assymetry for extra border image
                        _border_extra_border = getborder(extra_border_mask) # get border for extra border image
                        color_extra_border = get_irregularity_score(img_rgb, extra_border_mask) # get color for extra border image (almost the same as original image)

                        # return all computed data
                        yield img_id, assymetry_noise, _border_noise, color_noise, assymetry_contrast, _border_contrast, color_contrast, assymetry_extra_border, _border_extra_border, color_extra_border
                    else:
                        # data augmentation should only be done on melanoma images, so if not MEL we skip it
                        continue
                else: # if data augmentation should not be applied, we just compute the normal ABC features here
                    # check if hair should be removed
                    if self.hairless:
                        # apply the hair removal function with the optimal parameters found through cohens kappa
                        blackhat, tresh, img_rgb = removeHair(img_rgb, img_gray, kernel_size=5, threshold=10, radius=3)
                    # get a binary array of the annotated mask
                    manual_mask = get_binary_mask(mask_path)
                    # use the original image and the annotated mask to compute our own region growing mask
                    mask = get_mask(img_rgb, manual_mask)
                    # fallback if the mask could not be compute and therefore it's all black
                    if np.sum(mask) == 0:
                        continue

                    # COMPUTE ABC FEATURES:
                    assymetry = get_asymmetry(mask) # get asymmetry
                    _border = getborder(mask) # get border
                    color = get_irregularity_score(img_rgb, mask) # get color

                    # return the ABC features
                    yield filename, assymetry, _border, color

            # throw an exception if an error ocurred
            except Exception as e:
                print(f"Error with image: {os.path.basename(filename)}. Error is {e}")