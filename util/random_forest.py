import pandas as pd                      
import numpy as np                       
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.stats import norm             


def random_forest(filepath: str, feature_version: str, n_estimators: int = 100):
    df = pd.read_csv(filepath)

    feat_cols = [
    "feature A " + feature_version,  
    "feature B " + feature_version,   
    "feature C " + feature_version]

    X, y = df[feat_cols], df["Label: Is cancer"]                   

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, f1s, aucs = [], [], []            

    for train_idx, test_idx in kf.split(X, y):  
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        clf = RandomForestClassifier(       # 100 trees, class-balanced
            n_estimators = n_estimators,
            class_weight="balanced", #Gives more weight to cancer (lesser class) 
            random_state=0)
        clf.fit(X_tr, y_tr)                 

        y_pred = clf.predict(X_te)          
        y_prob = clf.predict_proba(X_te)[:, 1] 

        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred))
   
        aucs.append(roc_auc_score(y_te, y_prob))

    acc, f1 = np.mean(accs), np.mean(f1s)  
    auc = np.mean(aucs) if aucs else float("nan")
    std = np.std(aucs) if aucs else float("nan")

    
    if aucs:
        z = norm.ppf(0.975)
        ci = (round(auc - z*std/np.sqrt(len(aucs)), 3),
              round(auc + z*std/np.sqrt(len(aucs)), 3))
    else:
        ci = (float("nan"), float("nan"))

    return (f"Acc mean={acc:.3f}",
            f"F1 mean={f1:.3f}",
            f"AUC mean={auc:.3f}",
            f"AUC std={std:.3f}",
            f"AUC 95% CI={ci}")


print(random_forest(r"util\\structured_cancer_data.csv", "(v1)"))
