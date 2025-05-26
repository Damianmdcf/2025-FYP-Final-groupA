from util.random_forest import random_forest
from util.knn import knn
from pathlib import Path
import os


droot= Path("data")
file_path= droot / "train-baseline-data.csv"



# Random Forest
# for i in (100, 200, 300, 500, 1000):
#     result_df= random_forest(file_path, i)    
#     out_csv = droot / "result-baseline-1.csv"
#     result_df.to_csv(out_csv, mode='a', index=False, header=not os.path.exists(out_csv))

# KNN neighboors
for k in (1, 3, 5, 7):
    result_df= knn(file_path, k) 
    out_csv = droot / "result-baseline-1.csv"
    result_df.to_csv(out_csv, mode='a', index=False, header=not os.path.exists(out_csv))