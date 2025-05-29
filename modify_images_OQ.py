import os
import cv2
import numpy as np
import pandas as pd
from util.img_util import ImageDataLoader

# loads variables from the .env file
from dotenv import load_dotenv
load_dotenv()

images_path = os.getenv("IMAGE_DATA_URL_LOCAL") #local path to lesion images
mask_path = os.getenv("MASK_DATA_URL_LOCAL") #local path to lesion masks

if __name__ == "__main__":

    #initializes the image loader applying 3 image augmentation methods per image in the minority class 
    loader = ImageDataLoader(images_path, mask_path, hairless = None, augmentation = True)

    rows = [] #will store the extracted features per image


    #we set names for consistensy of the features in the code
    A = "feature_a"
    B = "feature_b"
    C = "feature_c"


    i = 1   #counter for loading process

    #iterate over all image samples that are by loaded by ImageDataLoader
    for img_id, assymetry_noise, _border_noise, color_noise, assymetry_contrast, _border_contrast, color_contrast, assymetry_extra_border, _border_extra_border, color_extra_border in loader:
        print(f"Now loading: {img_id}")
        #each minority class image is being augmented in tree ways, noise, contrast abd extra border
        #for each one of them, the features are calculated and saved
        rows.append({"img_id": f"NOISE_{img_id}", A: assymetry_noise, B: _border_noise, C: color_noise, "augmentation_type": "noise", "original_img_id": img_id})
        rows.append({"img_id": f"CONTRAST_{img_id}", A: assymetry_contrast, B: _border_contrast, C: color_contrast, "augmentation_type": "contrast", "original_img_id": img_id})
        rows.append({"img_id": f"EXTRA_BORDER_{img_id}", A: assymetry_extra_border, B: _border_extra_border, C: color_extra_border, "augmentation_type": "extra_border", "original_img_id": img_id})
        print(f"{i} done {42-i} to go")
        i += 1
    
    #Converts the list of dictionaries into a data frame
    df_features = pd.DataFrame(rows)
    
    #we standarized each feature for better scaling 
    df_features["Z_" + A] = (df_features[A] - df_features[A].mean()) / df_features[A].std()
    df_features["Z_" + B] = (df_features[B] - df_features[B].mean()) / df_features[B].std()
    df_features["Z_" + C] = (df_features[C] - df_features[C].mean()) / df_features[C].std()
    
    #All syntetic samples as label as melanoma
    df_features["Melanoma"] = 1

    # we load the baseline calculated features
    baseline_df = pd.read_csv("data/train-baseline-data.csv")

    # add synthetic data info
    baseline_df["original_img_id"] = baseline_df["img_id"] #reference for originals
    baseline_df["is_synthetic"] = False #marks them as not syntetic

    # adds a systentic "mark" to the augmented dataset
    df_features["is_synthetic"] = True

    # Makes sure column order is the same in the two df's
    column_order = [
    "img_id", "original_img_id", "Melanoma", 
    "feature_a", "feature_b", "feature_c", 
    "Z_feature_a", "Z_feature_b", "Z_feature_c", 
    "augmentation_type", "is_synthetic"
]
    df_features = df_features[column_order]


    # combine the two DataFrames (original and systetic)
    combined_df = pd.concat([baseline_df, df_features], ignore_index=True)


    # we save the new dataset to a new csv file for model training
    combined_df.to_csv("data/train-OQ-data-for-model.csv", index=False)