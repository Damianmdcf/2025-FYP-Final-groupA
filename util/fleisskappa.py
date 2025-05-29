import pandas as pd
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa

# Read the human annotation data from a CSV file
df = pd.read_csv("annotations/annotations.csv")
# Remove the 'img_id' column as it's not needed for Fleiss' Kappa calculation
df = df.drop(columns=["img_id"])
# Gets the number of unique rating categories (i.e. 3 in total. Rating 0 (no hair), rating 1 (some hair), rating 2 (a lot of hair))
categories = sorted(df.stack().unique())
# Initialize an empty matrix to store rating counts
ratings_matrix = np.zeros((len(df), len(categories)), dtype=int)
# Iterate over each item (row in the original df)
for i, row in enumerate(df.values):
    # Iterate over each unique category (i.e. 3 in total)
    for j, category in enumerate(categories): 
        # Count how many raters assigned the current rating to the current image
        # Create a boolean array, and count the true values
        ratings_matrix[i, j] = np.sum(row == category)

# Calculate Fleiss' Kappa using the ratings_matrix
kappa = fleiss_kappa(ratings_matrix)
# Print the calculated Fleiss' Kappa value
print("Fleiss' Kappa:", kappa)