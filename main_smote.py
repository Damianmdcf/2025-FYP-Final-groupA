import os
from util.testing import testing

if __name__ == "__main__":
    """
    Run this file to test your data on our SMOTE classifier from our open question.
    OBS: Make sure to run the main_baseline.py file first to compute the ABC features for your images.
    """
    if os.path.exists("data/final_baseline_features.csv"):
        # smote traning data path
        train_csv = "data/train-smote-data.csv"
        # your test data path computed by running main_baseline.py
        test_csv = "data/final_baseline_features.csv"
        # path where the results will be stored
        result_path = "result/final_smote"
        # run the SMOTE classifier on your test data
        testing(train_csv, test_csv, result_path, threshold=0.3)

        print("The predictions of the dataset has been saved in the folder 'result' under the name 'final_smote_predictions.csv'")
        print("The metrics of the dataset has been saved in the folder 'result' under the name 'final_smote_metrics.csv'")
    else:
        print("Make sure to run the main_baseline.py file first to compute the ABC features for your images.")
        print("The ABC features will then be saved in the data folder under the name 'data/final_baseline_features.csv'")
        print("Then you can run this file.")