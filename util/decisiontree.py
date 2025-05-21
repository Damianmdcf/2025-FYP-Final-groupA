# Decision-tree with 5-fold stratified CV on ABC features
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import norm


def decision_tree(filepath: str, feature_version: str):
    df = pd.read_csv(filepath)

    feat_cols = [
    "feature A " + feature_version,   # → "feature A (v1)"
    "feature B " + feature_version,   # → "feature B (v1)"
    "feature C " + feature_version]   # → "feature C (v1)"

    X = df[feat_cols]                       
    y = df["Label: Is cancer"]              

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, f1s, aucs = [], [], []

    for train_idx, test_idx in kf.split(X, y):         
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X_train, y_train)                      

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_prob))


    acc_mean, f1_mean = np.mean(accs), np.mean(f1s)
    auc_mean, auc_std = np.mean(aucs), np.std(aucs)

    z = norm.ppf(0.975)                                 # 95 % CI
    margin = z * auc_std / np.sqrt(kf.get_n_splits())
    ci = (round(auc_mean - margin, 3), round(auc_mean + margin, 3))

    return (
        f"Accuracy mean = {acc_mean:.3f}",
        f"F1 mean = {f1_mean:.3f}",
        f"AUC mean = {auc_mean:.3f}",
        f"AUC std = {auc_std:.3f}",
        f"AUC 95% CI = {ci}"
    )

print(decision_tree(r"util\test1.csv", "(v1)"))




