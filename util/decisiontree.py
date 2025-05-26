# Decision-tree with 5-fold stratified CV on ABC features
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import norm
import os
from sklearn.metrics import confusion_matrix


def decision_tree(filepath: str, feature_version: str, max_depth_given):
    df = pd.read_csv(filepath)

    feat_cols = ["Z_feature_a", "Z_feature_b", "Z_feature_c"]

    X = df[feat_cols]                       
    y = df["Melanoma"]              

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, f1s, aucs = [], [], []

    for train_idx, test_idx in kf.split(X, y):         
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = DecisionTreeClassifier(max_depth= max_depth_given, random_state=0)
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

    result_df = pd.DataFrame([{
        "Model": f"Decision_tree {feature_version}",
        "Accuracy Mean": round(acc_mean, 3),
        "F1 Mean": round(f1_mean, 3),
        "AUC Mean": round(auc_mean, 3),
        "AUC Std. Dev": round(auc_std, 3),
        "AUC 95% CI Lower": ci[0],
        "AUC 95% CI Upper": ci[1],
    }])

    result_csv_path = r"\data\result-baseline.csv"
    result_df.to_csv(result_csv_path, mode='a', index=False, header=not os.path.exists(result_csv_path))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return result_df, aucs

for d in range(1, 8):
    decision_tree(r"data\extended-data-for-model.csv", "V", d)