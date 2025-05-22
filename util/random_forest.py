import pandas as pd                      
import numpy as np                       
from sklearn.ensemble import RandomForestClassifier  
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from scipy.stats import norm             


def random_forest(filepath, feature_version, n_estimators: int = 100):
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
            n_estimators = n_estimators, #Gives more weight to cancer (lesser class)  bootstrap
            random_state=0)
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

    return pd.DataFrame([{
        "Model": f"Random_forest {feature_version},",
        "Accuracy Mean": round(acc, 3),
        "F1 Mean": round(f1, 3),
        "AUC Mean": round(auc_mean, 3),
        "AUC Std. Dev": round(std, 3),
        "AUC 95% CI": confidence_interval,
    }])

print(random_forest(r"util\structured_cancer_data.csv", "(v1)"))