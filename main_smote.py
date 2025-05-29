import os
from util.testing import testing
from dotenv import load_dotenv
load_dotenv()

images_path = os.getenv("IMAGE_DATA_URL_LOCAL")
mask_path = os.getenv("MASK_DATA_URL_LOCAL")

if __name__ == "__main__":
    if os.path.exists("data/final_baseline_features.csv"):
        train_csv = "data/train-smote-data.csv"
        test_csv = "data/final_baseline_features.csv"
        result_path = "result/final_smote"

        testing(train_csv, test_csv, result_path, threshold=0.3)

        print("The predictions of the dataset has been saved in the folder 'result' under the name 'final_smote_predictions.csv'")
        print("The metrics of the dataset has been saved in the folder 'result' under the name 'final_smote_metrics.csv'")
    else:
        print("Make sure to run the main_baseline.py file first to compute the ABC features for your images.")
        print("The ABC features will then be saved in the data folder under the name 'data/final_baseline_features.csv'")
        print("Then you can run this file.")
