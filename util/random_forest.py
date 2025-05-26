import pandas as pd                      
import numpy as np                       
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.stats import norm             
from sklearn.metrics import confusion_matrix
import os
from pathlib import Path


droot= Path("data")


fold_rows= []


def random_forest(filepath, feature_version,n_estimators: int = 100):
    df = pd.read_csv(filepath)

    feat_cols = ["Z_feature_a", "Z_feature_b", "Z_feature_c"]

    X, y = df[feat_cols], df["Melanoma"]                   

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, f1s, aucs = [], [], []            

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):  
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        clf = RandomForestClassifier(       # 100 trees, class-balanced
            max_depth=5, n_estimators = n_estimators, random_state=0)
        clf.fit(X_tr, y_tr)                 

        y_pred = clf.predict(X_te)          
        y_prob = clf.predict_proba(X_te)[:, 1] 

        acc = accuracy_score(y_te, y_pred)
        f1  = f1_score(y_te, y_pred)
        auc = roc_auc_score(y_te, y_prob)

        accs.append(acc);  f1s.append(f1);  aucs.append(auc)

        fold_rows.append({
            "Model": f"Random Forest {feature_version}",
            "Fold" : fold,
            "Accuracy": round(acc, 3),
            "F1":       round(f1, 3),
            "AUC":      round(auc, 3)
        })

    per_fold_path = droot / "result-fold-metrics-extended.csv"

    # wide = pd.DataFrame(fold_rows).pivot(index="Model", columns="Fold", values=["F1", "AUC"])
    # wide.columns = [f"{m} fold {f}" for m, f in wide.columns]      
    # wide.reset_index().to_csv(per_fold_path,
    #                        mode="a", index=False,
    #                        header=not os.path.exists(per_fold_path))

    acc, f1 = np.mean(accs), np.mean(f1s)  
    auc_mean = np.mean(aucs) if aucs else float("nan")
    std = np.std(aucs) if aucs else float("nan")

    
    if aucs:
        z = norm.ppf(0.975)
        confidence_interval = (round(auc_mean - z*std/np.sqrt(len(aucs)), 3),
              round(auc_mean + z*std/np.sqrt(len(aucs)), 3))
    else:
        confidence_interval = (float("nan"), float("nan"))

    result_df = pd.DataFrame([{
        "Model": f"Random_forest {feature_version},",
        "Accuracy Mean": round(acc, 3),
        "F1 Mean": round(f1, 3),
        "AUC Mean": round(auc_mean, 3),
        "AUC Std. Dev": round(std, 3),
        "AUC 95% CI": confidence_interval,
    }])
    # result_csv_path = droot / "result-hair-removal.csv"
    # result_df.to_csv(result_csv_path, mode='a', index=False, header=not os.path.exists(result_csv_path))

    print("Confusion Matrix:")
    print(confusion_matrix(y_te, y_pred))

    result_df, aucs

file_path= droot / "train-extended-data.csv"
random_forest(file_path, "v1000", 1000)