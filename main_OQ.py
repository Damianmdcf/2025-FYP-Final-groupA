import os
import cv2
import numpy as np
import pandas as pd
from util.img_util import ImageDataLoader
from dotenv import load_dotenv
load_dotenv()

images_path = os.getenv("IMAGE_DATA_URL_LOCAL")
mask_path = os.getenv("MASK_DATA_URL_LOCAL")

if __name__ == "__main__":
    loader = ImageDataLoader(images_path, mask_path, hairless = None, augmentation = True)

    rows = []

    A = "feature_a"
    B = "feature_b"
    C = "feature_c"


    i = 1
    for img_id, assymetry_noise, _border_noise, color_noise, assymetry_contrast, _border_contrast, color_contrast, assymetry_extra_border, _border_extra_border, color_extra_border in loader:
        print(f"Now loading: {img_id}")
        rows.append({"img_id": f"NOISE_{img_id}", A: assymetry_noise, B: _border_noise, C: color_noise, "augmentation_type": "noise", "original_img_id": img_id})
        rows.append({"img_id": f"CONTRAST_{img_id}", A: assymetry_contrast, B: _border_contrast, C: color_contrast, "augmentation_type": "contrast", "original_img_id": img_id})
        rows.append({"img_id": f"EXTRA_BORDER_{img_id}", A: assymetry_extra_border, B: _border_extra_border, C: color_extra_border, "augmentation_type": "extra_border", "original_img_id": img_id})
        print(f"{i} done {42-i} to go")
        i += 1
    
    df_features = pd.DataFrame(rows)
    df_features["Z_" + A] = (df_features[A] - df_features[A].mean()) / df_features[A].std()
    df_features["Z_" + B] = (df_features[B] - df_features[B].mean()) / df_features[B].std()
    df_features["Z_" + C] = (df_features[C] - df_features[C].mean()) / df_features[C].std()
    df_features["Melanoma"] = 1


    baseline_df = pd.read_csv("data/train-baseline-data.csv")

    # Add synthetic data info
    baseline_df["original_img_id"] = baseline_df["img_id"]
    baseline_df["is_synthetic"] = False

    # Add synthetic data info
    df_features["is_synthetic"] = True

    # Make sure column order is the same in the two df's
    column_order = [
    "img_id", "original_img_id", "Melanoma", 
    "feature_a", "feature_b", "feature_c", 
    "Z_feature_a", "Z_feature_b", "Z_feature_c", 
    "augmentation_type", "is_synthetic"
]
    df_features = df_features[column_order]


    # Combine the two DataFrames
    combined_df = pd.concat([baseline_df, df_features], ignore_index=True)

    combined_df.to_csv("data/new-train-OQ-data-for-model.csv", index=False)