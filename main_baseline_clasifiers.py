from util.random_forest import random_forest
from util.knn import knn
from util.decisiontree import decision_tree

from pathlib import Path
import os


droot= Path("data")
file_path= droot / "train-baseline-data.csv"
out_path= droot / "result-baseline-1.csv"

# Random Forest
for i in (100, 200, 300, 500, 1000):
    result_df_random= random_forest(file_path, i)    
    out_csv = out_path
    result_df_random.to_csv(out_csv, mode='a', index=False, header=not os.path.exists(out_csv))

# KNN
for k in (1, 3, 5, 7):
    result_df_knn= knn(file_path, k) 
    out_csv = out_path
    result_df_knn.to_csv(out_csv, mode='a', index=False, header=not os.path.exists(out_csv))

# Decision Tree
for d in range(1, 8):
    result_df_decition= decision_tree(file_path, d)
    out_csv = out_path
    result_df_decition.to_csv(out_csv, mode='a', index=False, header=not os.path.exists(out_csv))

# Logistic Regression
from util.logistic_regression import logistic_regression
for t in (0.5, 0.3, 0.03, 0.01):
    result_df_logistic= logistic_regression(file_path, t)
    out_csv = out_path
    result_df_logistic.to_csv(out_csv, mode='a', index=False, header=not os.path.exists(out_csv))


#Extended method (with hair removal)
droot= Path("data")
file_path_extended= droot / "train-extended-data.csv"
out_path_extended= droot / "result-extended-1.csv"

# Random Forest
for i in (100, 200, 300, 500, 1000):
    result_df_random= random_forest(file_path_extended, i)    
    out_csv = out_path_extended
    result_df_random.to_csv(out_csv, mode='a', index=False, header=not os.path.exists(out_csv))

# KNN
for k in (1, 3, 5, 7):
    result_df_knn= knn(file_path_extended, k) 
    out_csv = out_path_extended
    result_df_knn.to_csv(out_csv, mode='a', index=False, header=not os.path.exists(out_csv))

# Decision Tree
for d in range(1, 8):
    result_df_decition= decision_tree(file_path_extended, d)
    out_csv = out_path_extended
    result_df_decition.to_csv(out_csv, mode='a', index=False, header=not os.path.exists(out_csv))

# Logistic Regression
from util.logistic_regression import logistic_regression
for t in (0.5, 0.3, 0.03, 0.01):
    result_df_logistic= logistic_regression(file_path_extended, t)
    out_csv = out_path_extended
    result_df_logistic.to_csv(out_csv, mode='a', index=False, header=not os.path.exists(out_csv))

