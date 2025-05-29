import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
import matplotlib.pyplot as plt
from pathlib import Path

"""
Resampling of the data, should be perfomered after data augmentation as we have very little melanoma cases
I created two functions, one uses only SMOTE (creating synthethic samples by interpolating between real minority samples) 
and the other applies both SMOTE on minority class and undersampling on majority class, have to decide which one works better
"""

# path to the data file
droot = Path("data")
# get the abc features
df_abc= pd.read_csv(droot / "train-baseline-data.csv")

def clean_abc_data(df, feature_cols):
    """
    removing rows with missing values in any of the feature columns
    """
    df_clean = df.dropna(subset=feature_cols)
    return df_clean

df_clean = clean_abc_data(df_abc, ['Z_feature_a', 'Z_feature_b', 'Z_feature_c'])

def apply_smote_only(df, feature_cols, id_col, label_col, sampling_ratio=0.3, k_neighbors=5):
    """
    Oversamples the minority class using SMOTE (creating synthethic samples by interpolating between real minority samples)
    """
    X = df[feature_cols].values
    y = df[label_col].values
    ids = df[id_col].values

    # Appplying SMOTE, could be good to try different k_neighbours
    # Sampling strategy is oversample the minority class to half the size of the majority one
    smote = SMOTE(sampling_strategy=sampling_ratio, k_neighbors=k_neighbors, random_state=42)
    X_s, y_s = smote.fit_resample(X, y)

    # Create a data frame with added samples
    df_s = pd.DataFrame([{
    'img_id': ids[i] if i < len(ids) else f'synthetic_{i - len(ids)}',
    'Melanoma': y_s[i],
    'Z_feature_a': X_s[i][0],
    'Z_feature_b': X_s[i][1],
    'Z_feature_c': X_s[i][2]
    } for i in range(len(X_s))])

    return df_s


def apply_smote_and_undersample(df, feature_cols, id_col, label_col, smote_ratio=0.3, under_ratio=0.5, k_neighbors=5):
    """
    Applying SMOTE like in apply_smote_only function, but also does undersampling of non-melanoma cases
    """
    # Extract data
    X = df[feature_cols]
    y = df[label_col]
    ids = df[id_col].to_list()

    # Oversample the minority class using SMOTE, adds synenthis samples, #0.1
    smote = SMOTE(sampling_strategy=smote_ratio, k_neighbors=k_neighbors, random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    # Identify how many synthetic samples were added, neccessary to correctly label data, as this time we are not only adding new data but later randomly deleting cases from majority class
    n_original = len(df)
    n_synthetic = len(X_smote) - n_original

    # Create new IDs for synthetic samples
    new_ids = ids + [f"synthetic_{i}.jpg" for i in range(n_synthetic)]

    # Create new DataFrame 
    df_smote = pd.DataFrame(X_smote, columns=feature_cols)
    df_smote[label_col] = y_smote
    df_smote[id_col] = new_ids

    # Undersample the majority class, randomly removes non-melanoma cases, reduces the risk of overfitting to the overrepresented majority class
    rus = RandomUnderSampler(sampling_strategy=under_ratio, random_state=42)
    X_final, y_final = rus.fit_resample(X_smote, y_smote)

    final_indices= rus.sample_indices_
    df_final= df_smote.iloc[final_indices].copy()

    # Return the augmentanted dataframe
    return df_final
    
def apply_undersampling(df, feature_cols, id_col, label_col, sampling_ratio=0.5, k_neighbors=5):
    """
    Applying undersampling of the non-melnoma images
    """
    # Undersamples the majority class, ranodmly removes non-melanoma cases
    X = df[feature_cols].values
    y = df[label_col].values
    ids = df[id_col].values

    # Appplying Random under-sampling
    rus = RandomUnderSampler(sampling_strategy=sampling_ratio, k_neighbors=k_neighbors, random_state=42)
    X_s, y_s = rus.fit_resample(X, y)

    # Created a data frame with added samples
    df_s = pd.DataFrame([{
    'img_id': ids[i] if i < len(ids) else f'synthetic_{i - len(ids)}',
    'Melanoma': y_s[i],
    'Z_feature_a': X_s[i][0],
    'Z_feature_b': X_s[i][1],
    'Z_feature_c': X_s[i][2]
    } for i in range(len(X_s))])

    return df_s