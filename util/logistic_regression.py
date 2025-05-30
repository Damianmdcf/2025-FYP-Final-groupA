import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import norm
from pathlib import Path
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# get the path to the data folder
droot = Path("data")  

def logistic_regression(filepath, treshold, apply_smote= False, smote_ratio=0.3, k_neighbors=5, apply_undersampling= False, under_ratio=0.5):
    """
    Logistic regression classifier. Classic logistic regression used to predict melanoma cases based on traning or test data.
    """
    # Read the traning data
    df = pd.read_csv(filepath)
    # The features to be used
    feat_cols = ["Z_feature_a", "Z_feature_b", "Z_feature_c"]
    # If A, B and C features could not be computed we drop the image.
    df = df.dropna(subset=feat_cols + ["Melanoma"])
    # Get the feature values and the label
    X, y = df[feat_cols], df["Melanoma"]

    # Used later for groupKfolds
    groups = None
    
    # check if we have augmented images or not
    if "original_img_id" in df.columns: 
        # we have augmented images, so we do StratifiedGroupKFold to keep all "versions" of an image in the same fold
        kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        groups = df["original_img_id"]
        # print("Doing data augmentation")
    else:
        # no augmented images so we just run normal stratifiedKFold
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # used later to write results to csv
    rows = []

    #For each one of the folds, use the train and "test" data to return crucial values (aucs, f1s, etc) 
    for fold, (tr, te) in enumerate(kf.split(X, y, groups=groups), start=1):
        # get the validation and traning features and labels
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        # if synthetic data is present then remove it from the validation set to avoid overfitting
        if "is_synthetic" in df.columns:
            old_Xte = Xte
            val_mask = df.iloc[te]["is_synthetic"] == False
            Xte = Xte[val_mask]
            yte = yte[val_mask]
        
        #If apply_SMOTE = True, oversample train data using SMOTE, test data stays untouched
        if apply_smote:
            sm= SMOTE(sampling_strategy=smote_ratio, k_neighbors=k_neighbors,random_state= 42)
            Xtr, ytr= sm.fit_resample(Xtr, ytr)
        
        #If apply_undersampling = True, oundersample train data using Random Under sampling, test data stays untouched
        if apply_undersampling:
            rus = RandomUnderSampler(sampling_strategy=under_ratio, random_state=42)
            Xtr, ytr= rus.fit_resample(Xtr, ytr)

        # Create a Logistic Regression classifier with the given max depth, train and fit one per fold 
        clf = LogisticRegression()
        # Fit on traning data
        clf.fit(Xtr, ytr)

        # Get the predicted probability of the positive class for each test sample
        yprob = clf.predict_proba(Xte)[:, 1]
        # Classify as 1 if probability > threshold, else 0
        ypred = (yprob > float(treshold)).astype(int)
        
        # Appends all the crucial values to a list so we can retrieve them later 
        rows.append({
            "Model": f"Logistic Regression {treshold}",
            "Fold":  fold,
            "Accuracy": accuracy_score(yte, ypred),
            "F1":       f1_score(yte, ypred),
            "AUC":      roc_auc_score(yte, yprob)
        })
        
    # Transforms values into data a frame
    per_fold = pd.DataFrame(rows).pivot( index="Model", columns="Fold", values=["Accuracy", "F1", "AUC"])
    per_fold.columns = [f"{metric}_fold_{fold}" for metric, fold in per_fold.columns]
    per_fold = per_fold.reset_index()
    
    # compute summary statistics
    n_folds = 5
    # mean for all the crucial values 
    per_fold["Accuracy Mean"]= per_fold[[f"Accuracy_fold_{i}" for i in range(1, n_folds+1)]].mean(axis=1)
    per_fold["F1 Mean"] = per_fold[[f"F1_fold_{i}"       for i in range(1, n_folds+1)]].mean(axis=1)
    per_fold["AUC Mean"]= per_fold[[f"AUC_fold_{i}"      for i in range(1, n_folds+1)]].mean(axis=1)
    # std dev of AUC
    per_fold["AUC Std. Dev"]= per_fold[[f"AUC_fold_{i}" for i in range(1, n_folds+1)]].std(axis=1)
    # 95% CI for AUC mean
    z = norm.ppf(0.975)
    per_fold["AUC 95% CI"]= per_fold.apply(lambda row: ( round(row["AUC Mean"] - z * row["AUC Std. Dev"] / np.sqrt(n_folds), 3), round(row["AUC Mean"] + z * row["AUC Std. Dev"] / np.sqrt(n_folds), 3)), axis=1)
    
    return per_fold