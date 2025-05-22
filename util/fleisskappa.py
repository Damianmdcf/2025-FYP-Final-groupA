import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa

df = pd.read_csv("../annotations/annotations.csv")

df = df.drop(columns=["img_id"])

categories = sorted(df.stack().unique())

ratings_matrix = np.zeros((len(df), len(categories)), dtype=int)

for i, row in enumerate(df.values):
    for j, category in enumerate(categories):
        ratings_matrix[i, j] = np.sum(row == category)

kappa = fleiss_kappa(ratings_matrix)
print("Fleiss' Kappa:", kappa)