import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
import os
from pathlib import Path

droot = Path("data")  

def logistic_regression(filepath, treshold):
    
    df = pd.read_csv(filepath)
    feat_cols = ["Z_feature_a", "Z_feature_b", "Z_feature_c"]
    df = df.dropna(subset=feat_cols + ["Melanoma"])  # Drop rows with NaN in specified columns
    X= df[feat_cols]
    y = df["Melanoma"]

    #Only uses the estandarized values 
    feat_cols = ["Z_feature_a", "Z_feature_b", "Z_feature_c"]

    X, y      = df[feat_cols], df["Melanoma"]
    
    rows = []

    #Apply K folds top the data 
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    #For each one of the folds, use the train and "test" data to return crucial values (aucs, f1s, etc) 
    for fold, (tr, te) in enumerate(kf.split(X, y), start=1):

        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        #Starts a Decision tree classifier with the given max depth, train andfit one per fold 

        clf = LogisticRegression()
        clf.fit(Xtr, ytr)


        yprob = clf.predict_proba(Xte)[:, 1]  # Get the predicted probability of the positive class for each test sample
        ypred = (yprob > float(treshold)).astype(int)  # Classify as 1 if probability > threshold, else 0
        
        #Appends all the crucial values to a list so we can retrieve them later 
        rows.append({
            "Model": f"Logistic Regression {treshold}",
            "Fold":  fold,
            "Accuracy": accuracy_score(yte, ypred),
            "F1":       f1_score(yte, ypred),
            "AUC":      roc_auc_score(yte, yprob)
        })

        print(f"Confusion Matrix, Logistic Regression {treshold}")
        print(confusion_matrix(yte, ypred))
        
    
    
    #Transforms values into data a frame
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