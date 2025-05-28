import os
import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

images_path = os.getenv("IMAGE_DATA_URL_LOCAL")
mask_path = os.getenv("MASK_DATA_URL_LOCAL")

from .inpaint_util import removeHair
from .mask import get_mask
from .feature_A import get_asymmetry
from .feature_B import getborder
from .feature_C import get_irregularity_score

from .augmentation import apply_clahe, roughen_border, noise_augmentation # REMEMBER TO FIX

def readImageFile(file_path):
    # read image as an 8-bit array
    img_bgr = cv2.imread(file_path)

    # convert to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # convert the original image to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    return img_rgb, img_gray

def get_binary_mask(mask_path):
    if mask_path:
        mask_img_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        _, img_th1 = cv2.threshold(mask_img_gray, 120, 255, cv2.THRESH_BINARY)
        return img_th1
    else:
        return None

def find_mask(image_path, mask_list, masks_filepath):
    basename = os.path.basename(image_path).replace(".png", "_mask.png")
    mask_path = masks_filepath + "/" + basename
    if mask_path in mask_list:
        return mask_path
    else:
        return None

def saveImageFile(img_rgb, file_path):
    try:
        # convert BGR
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # save the image
        success = cv2.imwrite(file_path, img_bgr)
        if not success:
            print(f"Failed to save the image to {file_path}")
        return success

    except Exception as e:
        print(f"Error saving the image: {e}")
        return False
    
            
class ImageDataLoader:
    def __init__(self, directory, masks, hairless=None, augmentation=None):
        self.directory = directory
        self.masks = masks
        self.hairless = hairless
        self.augmentation = augmentation
        self.file_list = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', 'jpeg', '.bmp', '.tiff'))])
        self.mask_list = sorted([os.path.join(masks, f) for f in os.listdir(masks) if f.lower().endswith(('.png', '.jpg', 'jpeg', '.bmp', '.tiff'))])
        if not self.file_list:
            raise ValueError("No image files found in the directory.")
        self.num_batches = len(self.file_list)

        self.df = pd.read_csv("data/metadata.csv")
        self.test_data = pd.read_csv('data/test-baseline-data.csv')
        self.test_mel = list(self.test_data[self.test_data["Melanoma"] == 1]["img_id"])

    def is_melanoma(self, image_id):
        diagnostic = self.df.loc[self.df['img_id'] == image_id, "diagnostic"].iloc[0]
        
        if diagnostic == "MEL" and image_id not in self.test_mel:
            return True
        else:
            return False

    def __len__(self):
        return self.num_batches
    
    def one_image(self, img_id):
        filename = self.directory + "/" + img_id
        mask_path = find_mask(filename, self.mask_list, self.masks)
        img_rgb, img_gray = readImageFile(filename)

        if self.hairless:
            blackhat, tresh, img_rgb = removeHair(img_rgb, img_gray)

        manual_mask = get_binary_mask(mask_path)

        mask = get_mask(img_rgb, manual_mask)

        # Compute the features
        assymetry = get_asymmetry(mask)
        _border = getborder(mask)
        color = get_irregularity_score(img_rgb, mask)

        return assymetry, _border, color, mask

    def __iter__(self):
        for filename in self.file_list:
            try: 
                mask_path = find_mask(filename, self.mask_list, self.masks)
                img_rgb, img_gray = readImageFile(filename)

                if self.augmentation:
                    img_id = os.path.basename(filename)
                    mel = self.is_melanoma(img_id)

                    if mel:
                    
                        manual_mask = get_binary_mask(mask_path)
                        mask = get_mask(img_rgb, manual_mask)

                        # Apply augmentation method to the img
                        noise_img = noise_augmentation(img_rgb)
                        noise_mask = mask

                        contrast_img = apply_clahe(img_rgb)
                        contrast_mask = get_mask(contrast_img, manual_mask)
                        if np.sum(contrast_mask) == 0:
                            contrast_mask = mask  # fallback if the mask is all black

                        # Save the new images on a different folder/path 
                        # dir_path = os.path.dirname(filename)
                        # new_dir = os.path.join(dir_path, "New")
                        # os.makedirs(new_dir, exist_ok=True)
                        # saveImageFile(img_rgb, os.path.join(new_dir, os.path.basename(filename)))
                        # mask = (mask.astype(np.uint8)) * 255
                        # cv2.imwrite(os.path.join(new_dir, "MASK_" + os.path.basename(filename)), mask)

                        extra_border_mask = roughen_border(mask)
                        if np.sum(extra_border_mask) == 0:
                            extra_border_mask = mask  # fallback

                  
                        # Compute the features for the augmented pictures
                        assymetry_noise = get_asymmetry(noise_mask)
                        _border_noise = getborder(noise_mask)
                        color_noise = get_irregularity_score(noise_img, noise_mask)

                        assymetry_contrast = get_asymmetry(contrast_mask)
                        _border_contrast = getborder(contrast_mask)
                        color_contrast = get_irregularity_score(contrast_img, contrast_mask)

                        assymetry_extra_border = get_asymmetry(extra_border_mask.astype(bool))
                        _border_extra_border = getborder(extra_border_mask)
                        color_extra_border = get_irregularity_score(img_rgb, extra_border_mask)

                        yield img_id, assymetry_noise, _border_noise, color_noise, assymetry_contrast, _border_contrast, color_contrast, assymetry_extra_border, _border_extra_border, color_extra_border

                    else:
                        continue
                else:
                    if self.hairless:
                        blackhat, tresh, img_rgb = removeHair(img_rgb, img_gray, kernel_size=5, threshold=10, radius=3)

                    manual_mask = get_binary_mask(mask_path)

                    mask = get_mask(img_rgb, manual_mask)

                    # Skip bad images where we couldn't compute a mask
                    if np.sum(mask) == 0:
                        continue             

                    # Compute the features
                    assymetry = get_asymmetry(mask)
                    _border = getborder(mask)
                    color = get_irregularity_score(img_rgb, mask)

                    yield filename, assymetry, _border, color


            except Exception as e:
                print(f"Error with image: {os.path.basename(filename)}. Error is {e}")