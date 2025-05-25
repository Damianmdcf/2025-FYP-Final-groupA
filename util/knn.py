import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
import os


def knn(file_path, feature_version, k):
    
    df = pd.read_csv(file_path)

    feat_cols = ["Z_feature_a", "Z_feature_b", "Z_feature_c"]
    df = df.dropna(subset=feat_cols + ["Melanoma"])  # Drop rows with NaN in specified columns
    features = df[feat_cols]
    labels = df["Melanoma"]

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, f1s, aucs = [], [], []

    for train_index, test_index in kf.split(features, labels):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        model = KNeighborsClassifier(n_neighbors=k)

        # Skip fold if only one class is present in y_train or y_test
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print("Skipping fold due to only one class in y_train or y_test.")
            continue

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_probs = model.predict_proba(X_test)[:, 1]

        accs.append(accuracy_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
        aucs.append(roc_auc_score(y_test, y_probs))

    accuracy = np.mean(accs)
    F1 = np.mean(f1s)
    auc_mean = np.mean(aucs)
    std_dev = np.std(aucs)

    z = norm.ppf(0.975)
    margin = z * std_dev / np.sqrt(5)
    confidence_interval = (round(auc_mean - margin, 3), round(auc_mean + margin,3))

    result_df= pd.DataFrame([{
        "Model": f"knn {feature_version}, (k={k})",
        "Accuracy Mean": round(accuracy, 3),
        "F1 Mean": round(F1, 3),
        "AUC Mean": round(auc_mean, 3),
        "AUC Std. Dev": round(std_dev, 3),
        "AUC 95% CI": confidence_interval }])

    result_csv_path = r"C:\Users\bruda\OneDrive\Escritorio\Projects\2025-FYP-Final-groupA\data\result-baseline.csv"
    result_df.to_csv(result_csv_path, mode='a', index=False, header=not os.path.exists(result_csv_path))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return result_df, aucs

for k in (3, 5, 7):
    knn(r"data/train-baseline-data.csv", "V", k)