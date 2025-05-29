from util.random_forest import random_forest
from util.knn import knn
from util.decisiontree import decision_tree

from pathlib import Path
import os


def evaluate_all_models(file_path, out_path,  apply_smote= False, smote_ratio=0.3, k_neighbors=5, apply_undersampling= False, under_ratio=0.5):
    """
    Run all classifiers on the traning data, to find the best model.
    """  
    # Random forest on various number of trees
    for i in (100, 200, 300, 500, 1000):
        result_df_random= random_forest(file_path, i,  apply_smote, smote_ratio, k_neighbors, apply_undersampling, under_ratio)    
        out_csv = out_path
        result_df_random.to_csv(out_csv, mode='a', index=False, header=not os.path.exists(out_csv))

    # KNN 
    for k in (1, 3, 5, 7):
        result_df_knn= knn(file_path, k,apply_smote, smote_ratio, k_neighbors, apply_undersampling, under_ratio ) 
        out_csv = out_path
        result_df_knn.to_csv(out_csv, mode='a', index=False, header=not os.path.exists(out_csv))

    # Decision Tree
    for d in range(1, 8):
        result_df_decition= decision_tree(file_path, d, apply_smote, smote_ratio, k_neighbors, apply_undersampling, under_ratio)
        out_csv = out_path
        result_df_decition.to_csv(out_csv, mode='a', index=False, header=not os.path.exists(out_csv))

    # Logistic Regression
    from util.logistic_regression import logistic_regression
    for t in (0.5, 0.3, 0.03, 0.01):
        result_df_logistic= logistic_regression(file_path, t, apply_smote, smote_ratio, k_neighbors, apply_undersampling, under_ratio)
        out_csv = out_path
        result_df_logistic.to_csv(out_csv, mode='a', index=False, header=not os.path.exists(out_csv))


# # Evaluate for all datasets [baseline, extended, augmented]
droot = Path("data")
# datasets = [
#     ("train-baseline-data.csv", "result-baseline-1.csv"),
#     ("train-extended-data.csv", "result-extended-1.csv")
# ]

# for in_file, out_file in datasets:
#     file_path = droot / in_file
#     out_path = droot / out_file
#     evaluate_all_models(file_path, out_path)

datasets_OQ = [
    ("train-OQ-data-for-model.csv", "result-OQ1.csv")]

for in_file, out_file in datasets_OQ:
    file_path = droot / in_file
    out_path = droot / out_file
    evaluate_all_models(file_path, out_path, apply_smote= True, smote_ratio=0.3, apply_undersampling= False)

