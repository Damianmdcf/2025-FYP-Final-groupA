import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
import os

def logistic_regression(file_path, feature_version):
    df = pd.read_csv(file_path)

    feat_cols = ["Z_feature_a", "Z_feature_b", "Z_feature_c"]
    df= df.dropna(subset=feat_cols + ["Melanoma"]) 


    features = df[feat_cols]
    labels = df["Melanoma"]

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, f1s, aucs = [], [], []

    for train_index, test_index in kf.split(features, labels):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs > 0.01).astype(int)

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_probs))

    accuracy = np.mean(accs)
    F1 = np.mean(f1s)
    auc_mean = np.mean(aucs)
    std_dev = np.std(aucs)

    z = norm.ppf(0.975)
    margin = z * std_dev / np.sqrt(kf.get_n_splits())
    confidence_interval = (round(auc_mean - margin, 3), round(auc_mean + margin, 3))

    result_df = pd.DataFrame([{
        "Model": f"logistic regression {feature_version}",
        "Accuracy Mean": round(accuracy, 3),
        "F1 Mean": round(F1, 3),
        "AUC Mean": round(auc_mean, 3),
        "AUC Std. Dev": round(std_dev, 3),
        "AUC 95% CI": confidence_interval
        }])
    
    result_csv_path = r"C:\Users\bruda\OneDrive\Escritorio\Projects\2025-FYP-Final-groupA\data\result-baseline.csv"
    result_df.to_csv(result_csv_path, mode='a', index=False, header=not os.path.exists(result_csv_path))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return result_df, aucs


logistic_regression(r"data/train-baseline-data.csv", "V1: 0.01")