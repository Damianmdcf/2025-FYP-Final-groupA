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
        rows.append({"img_id": f"{img_id}_noise", A: assymetry_noise, B: _border_noise, C: color_noise})
        rows.append({"img_id": f"{img_id}_contrast", A: assymetry_contrast, B: _border_contrast, C: color_contrast})
        rows.append({"img_id": f"{img_id}_extra_border", A: assymetry_extra_border, B: _border_extra_border, C: color_extra_border})
        print(f"{i} done {52-i} to go")
        if i == 4:
            break
        i += 1
    
    df_features = pd.DataFrame(rows)
    df_features["Z_" + A] = (df_features[A] - df_features[A].mean()) / df_features[A].std()
    df_features["Z_" + B] = (df_features[B] - df_features[B].mean()) / df_features[B].std()
    df_features["Z_" + C] = (df_features[C] - df_features[C].mean()) / df_features[C].std()
    df_features["Melanoma"] = 1


    baseline_df = pd.read_csv("data/baseline-data-for-model.csv")

    # Make sure column order is the same
    column_order = ["img_id", "Melanoma", "feature_a", "feature_b", "feature_c", "Z_feature_a", "Z_feature_b", "Z_feature_c"]
    df_features = df_features[column_order]

    # Combine the two DataFrames
    combined_df = pd.concat([baseline_df, df_features], ignore_index=True)

    combined_df.to_csv("data/OQ-data-for-model.csv", index=False)