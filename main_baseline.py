import os
import pandas as pd
from util.img_util import ImageDataLoader
from util.testing import testing
from dotenv import load_dotenv
load_dotenv()

images_path = os.getenv("IMAGE_DATA_URL_LOCAL")
mask_path = os.getenv("MASK_DATA_URL_LOCAL")

if __name__ == "__main__":
    ### COMPUTE ABC FEATURES:
    loader = ImageDataLoader(images_path, mask_path, hairless = False)

    df = pd.DataFrame(columns=["img_id", "Melanoma"])
    metadata = pd.read_csv("data/metadata.csv")
    df["img_id"] = metadata["img_id"]
    df["Melanoma"] = metadata["diagnostic"]

    cancer = ["MEL"]
    df["Melanoma"] = df["Melanoma"].isin(cancer).astype(int)

    rows = []

    A = "feature_a"
    B = "feature_b"
    C = "feature_c"

    i = 1
    for filename, assymetry, _border, color in loader:
        img_id = os.path.basename(filename)
        print(f"Now loading: {img_id}")
        rows.append({"img_id": img_id, A: assymetry, B: _border, C: color})
        print(f"{i} done {len(loader.file_list)-i} to go")
        if i == 5:
            break
        i += 1
    
    df_features = pd.DataFrame(rows)
    df_features["Z_" + A] = (df_features[A] - df_features[A].mean()) / df_features[A].std()
    df_features["Z_" + B] = (df_features[B] - df_features[B].mean()) / df_features[B].std()
    df_features["Z_" + C] = (df_features[C] - df_features[C].mean()) / df_features[C].std()

    df_merged = pd.merge(df, df_features, on="img_id", how="outer")

    # Store the ABC features
    test_data_path = "data/final_baseline_features.csv"
    df_merged.to_csv(test_data_path, index=False)
    print(f"Your features has been saved in the data folder under the name '{test_data_path}'")

    testing("data/train-baseline-data.csv", test_data_path, "result/final_baseline", threshold=0.03)

    print("The predictions of the dataset has been saved in the folder 'result' under the name 'final_baseline_predictions.csv'")
    print("The metrics of the dataset has been saved in the folder 'result' under the name 'final_baseline_metrics.csv'")