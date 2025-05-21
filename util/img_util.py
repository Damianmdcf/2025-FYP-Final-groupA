
import random
import os
import cv2
import numpy as np
from dotenv import load_dotenv
load_dotenv()

images_path = os.getenv("IMAGE_DATA_URL_LOCAL")

from inpaint_util import removeHair
from feature_A import get_asymmetry
from feature_B import get_compactness
from feature_B import border

results = [] 

def readImageFile(file_path):
    # read image as an 8-bit array
    img_bgr = cv2.imread(file_path)

    # convert to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # convert the original image to grayscale
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    return img_rgb, img_gray


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
    def __init__(self, directory, shuffle=False, transform=None):
        self.directory = directory
        self.shuffle = shuffle
        self.transform = transform

        self.file_list= sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', 'jpeg', '.bmp', '.tiff'))])

        if not self.file_list:
            raise ValueError("No image files found in the directory.")

        # shuffle file list if required
        if self.shuffle:
            random.shuffle(self.file_list)

        # get the total number of batches
        self.num_batches = len(self.file_list)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        #Iterating throught all the images and applying transformations if necessesary"
        for filename in self.file_list:
            img_rgb, img_gray = readImageFile(filename)
            blackhat, tresh, img_hairless = removeHair(img_rgb, img_gray)

            #Save the new images on a different folder/path 
            dir_path = os.path.dirname(filename)
            new_dir = os.path.join(dir_path, "New")
            os.makedirs(new_dir, exist_ok=True)
            saveImageFile(img_hairless, os.path.join(new_dir, os.path.basename(filename)))

            # Get the mask
            img_mask_hairless = mask(img_hairless, threshold=25)

            # Compute the features
            assymetry = get_asymmetry(img_mask_hairless)
            # compactness = get_compactness(img_mask_hairless)
            _border = border(tresh)

            yield img_rgb, img_gray, filename, assymetry, compactness, _border
            # yield img_rgb, img_gray, filename, compactness, _border


current_directory = os.path.dirname(os.path.abspath(__file__))
relative_path_to_data = os.path.join(current_directory, images_path)
data_folder_path = os.path.normpath(relative_path_to_data)

loader = ImageDataLoader(images_path)

i = 0
for img_rgb, img_gray, filename, assymetry, compactness, _border in loader:                
    print(f"{filename:25s}  Assymetry = {assymetry:.4f}  Compactness = {compactness:.4f}   Border = {_border:.4f}")
    i += 1
    if i == 4:
        break