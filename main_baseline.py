import os
import pandas as pd


import matplotlib.pyplot as plt
from util.feature_A import crop


from util.img_util import ImageDataLoader
from dotenv import load_dotenv
load_dotenv()

images_path = os.getenv("IMAGE_DATA_URL_LOCAL")
mask_path = os.getenv("MASK_DATA_URL_LOCAL")

if __name__ == "__main__":
    loader = ImageDataLoader(images_path, mask_path, hairless = False)
    # error_image = "PAT_1022_115_132.png"
    # assymetry, _border, color, mask = loader.one_image(error_image)
    # cropped = crop(mask)
    # plt.imshow(mask)
    # plt.title("Mask Preview")
    # plt.axis('off')  # optional: hides axis ticks
    # plt.show()


    df = pd.DataFrame(columns=["img_id", "Label"])
    metadata = pd.read_csv("data/metadata.csv")
    df["img_id"] = metadata["img_id"]
    df["Melanoma"] = metadata["diagnostic"]

    cancer = ["MEL"]
    df["Melanoma"] = df["Melanoma"].isin(cancer).astype(int)

    # print("Please input feature version suffix:")
    # version_suffix = input()

    rows = []

    A = "feature_a"
    B = "feature_b"
    C = "feature_c"

    i = 1
    for filename, assymetry, _border, color in loader:
        img_id = os.path.basename(filename)
        print(f"Now loading: {img_id}")
        rows.append({"img_id": img_id, A: assymetry, B: _border, C: color})
        print(f"{i} done {2297-i} to go")
        i += 1
    
    df_features = pd.DataFrame(rows)
    df_features["Z_" + A] = (df_features[A] - df_features[A].mean()) / df_features[A].std()
    df_features["Z_" + B] = (df_features[B] - df_features[B].mean()) / df_features[B].std()
    df_features["Z_" + C] = (df_features[C] - df_features[C].mean()) / df_features[C].std()

    df_merged = pd.merge(df, df_features, on="img_id", how="outer")
    df_merged.to_csv("data/baseline-data-for-model.csv", index=False)