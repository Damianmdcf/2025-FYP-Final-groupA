import pandas as pd                       
import numpy as np                       
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from scipy.stats import norm             
from pathlib import Path


droot = Path("data")  

def random_forest(filepath, n_estimators: int = 100):
    
    df = pd.read_csv(filepath)

    #Only uses the estandarized values 
    feat_cols = ["Z_feature_a", "Z_feature_b", "Z_feature_c"]

    X, y = df[feat_cols], df["Melanoma"]

    groups = None

    # check if we have augmented images or not
    if "original_img_id" in df.columns: 
        # we have augmented images, so we do StratifiedGroupKFold to keep all "versions" of an image in the same fold
        kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        groups = df["original_img_id"]
    else:
        # no augmented images so we just run normal stratifiedKFold
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []

    #For each one of the folds, use the train and "test" data to return crucial values (aucs, f1s, etc) 
    for fold, (tr, te) in enumerate(kf.split(X, y, groups=groups), start=1):

        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        # if synthetic data is present then remove it from the validation set to avoid overfitting
        if "is_synthetic" in df.columns:
            val_mask = df.iloc[te]["is_synthetic"] == False
            Xte = Xte[val_mask]
            yte = yte[val_mask]

        #Starts a Random forest classifier, train and fit one per fold 
        clf = RandomForestClassifier( max_depth=5, n_estimators=n_estimators, random_state=0)
        clf.fit(Xtr, ytr)

        ypred = clf.predict(Xte)
        yprob = clf.predict_proba(Xte)[:,1]
        
        #Appends all the crucial values to a list so we can retrieve them later 
        rows.append({
            "Model": f"Random Forest {n_estimators}",
            "Fold":  fold,
            "Accuracy": accuracy_score(yte, ypred),
            "F1":       f1_score(yte, ypred),
            "AUC":      roc_auc_score(yte, yprob)
        })

        print(f"Confusion Matrix, Random Forest {n_estimators}")
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


