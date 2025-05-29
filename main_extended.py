import os
import pandas as pd
from util.img_util import ImageDataLoader
from util.testing import testing

# import dotenv to load .env paths
from dotenv import load_dotenv
load_dotenv()

# Get the path to where on your local machine images and masks lie.
# OBS: Remeber to create .env file as explained in README file.
images_path = os.getenv("IMAGE_DATA_URL_LOCAL")
mask_path = os.getenv("MASK_DATA_URL_LOCAL")

if __name__ == "__main__":
    """
    Run this file to test your data on our extended classifier.
    """
    # Construct data loader object to read, transform and loop over all images 
    loader = ImageDataLoader(images_path, mask_path, hairless = True)
    # Create a data frame with just the img_id and a binary label for melanoma or not
    df = pd.DataFrame(columns=["img_id", "Melanoma"])

    # OBS: As described in the README file, you should make sure to modify/replace the data/metadata.csv file
    # with the actual labels for your testing data. See README file if in doubt.
    metadata = pd.read_csv("data/metadata.csv")
    df["img_id"] = metadata["img_id"]
    df["Melanoma"] = metadata["diagnostic"]

    cancer = ["MEL"]
    df["Melanoma"] = df["Melanoma"].isin(cancer).astype(int)

    # used to store the data before loading to dataframe
    rows = []

    # name of the columns in the training data csv file
    A = "feature_a"
    B = "feature_b"
    C = "feature_c"

    # loop over all images in the images_path and compute the ABC features with hair removed
    i = 1
    for filename, assymetry, _border, color in loader:
        # get the img_id from the filename
        img_id = os.path.basename(filename)
        print(f"Now loading: {img_id}")
        # add the ABC features to the rows list
        rows.append({"img_id": img_id, A: assymetry, B: _border, C: color})
        print(f"{i} done {len(loader.file_list)-i} to go")
        i += 1

    # Create a dataframe with all feature data
    df_features = pd.DataFrame(rows)
    # Add a scaled Z-score version of each feature how avoid feature scale issues in classifiers
    df_features["Z_" + A] = (df_features[A] - df_features[A].mean()) / df_features[A].std()
    df_features["Z_" + B] = (df_features[B] - df_features[B].mean()) / df_features[B].std()
    df_features["Z_" + C] = (df_features[C] - df_features[C].mean()) / df_features[C].std()

    # Merge the dataframes to have features and labels together
    df_merged = pd.merge(df, df_features, on="img_id", how="outer")

    # Store the dataframe with the newly computed ABC features and label
    test_data_path = "data/final_extended_features.csv"
    df_merged.to_csv(test_data_path, index=False)
    print(f"Your features has been saved in the data folder under the name '{test_data_path}'")

    # Using the newly computed ABC features and label, test the model on our extended classifier
    testing("data/train-extended-data.csv", test_data_path, "result/final_extended", threshold=0.03)

    print("The predictions of the dataset has been saved in the folder 'result' under the name 'final_extended_predictions.csv'")
    print("The metrics of the dataset has been saved in the folder 'result' under the name 'final_extended_metrics.csv'")