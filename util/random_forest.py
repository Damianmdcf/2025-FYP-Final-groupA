import pandas as pd                      
import numpy as np                       
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.stats import norm             
from sklearn.metrics import confusion_matrix
import os


def random_forest(filepath, feature_version,n_estimators: int = 100):
    df = pd.read_csv(filepath)

    feat_cols = ["Z_feature_a", "Z_feature_b", "Z_feature_c"]

    X, y = df[feat_cols], df["Melanoma"]                   

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, f1s, aucs = [], [], []            

    for train_idx, test_idx in kf.split(X, y):  
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        clf = RandomForestClassifier(       # 100 trees, class-balanced
            n_estimators = n_estimators, random_state=0)
        clf.fit(X_tr, y_tr)                 

        y_pred = clf.predict(X_te)          
        y_prob = clf.predict_proba(X_te)[:, 1] 

        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred))
   
        aucs.append(roc_auc_score(y_te, y_prob))

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
    result_csv_path = r"C:\Users\bruda\OneDrive\Escritorio\Projects\2025-FYP-Final-groupA\data\result-baseline.csv"
    result_df.to_csv(result_csv_path, mode='a', index=False, header=not os.path.exists(result_csv_path))

    print("Confusion Matrix:")
    print(confusion_matrix(y_te, y_pred))

    result_df, aucs


random_forest(r"data/train-baseline-data.csv", "v300", 300)