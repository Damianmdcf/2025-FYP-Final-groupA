import os
import pandas as pd
from util.img_util import ImageDataLoader
from dotenv import load_dotenv
load_dotenv()

images_path = os.getenv("IMAGE_DATA_URL_LOCAL")
mask_path = os.getenv("MASK_DATA_URL_LOCAL")

if __name__ == "__main__":
    loader = ImageDataLoader(images_path, mask_path, hairless = True)
    df = pd.DataFrame(columns=["img_id", "Label"])
    metadata = pd.read_csv("data/metadata.csv")
    df["img_id"] = metadata["img_id"]
    df["Label"] = metadata["diagnostic"]

    cancer = ["BCC", "SCC", "MEL"]
    df["Label"] = df["Label"].isin(cancer).astype(int)

    print("Please input feature version suffix:")
    version_suffix = input()

    rows = []
    i = 0
    A = "feature_a_" + version_suffix
    B = "feature_b_" + version_suffix
    C = "feature_c_" + version_suffix

    for filename, assymetry, _border, color in loader:
        img_id = os.path.basename(filename)
        rows.append({"img_id": img_id, A: assymetry, B: _border, C: color})
        i += 1
        if i == 4:
            break
    
    df_features = pd.DataFrame(rows)
    df_features["Z_" + A] = (df_features[A] - df_features[A].mean()) / df_features[A].std()
    df_features["Z_" + B] = (df_features[B] - df_features[B].mean()) / df_features[B].std()
    df_features["Z_" + C] = (df_features[C] - df_features[C].mean()) / df_features[C].std()

    df_merged = pd.merge(df, df_features, on="img_id", how="outer")
    df_merged.to_csv("data/extended-data-for-model.csv", index=False)