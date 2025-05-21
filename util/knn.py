import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import KFold
from scipy.stats import norm


def knn(file_path, k):
    
    df = pd.read_csv(file_path)


    features = df[["feature A (v1)", "feature B (v1)", "feature C (v1)"]]
    labels = df["Label: Is cancer"]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    accs, f1s, aucs = [], [], []

    for train_index, test_index in kf.split(features):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        model = KNeighborsClassifier(n_neighbors=k)
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

    return f"K = {k}", f"Accuracy Mean = {round(accuracy, 3)}", f"F1 Mean= {round(F1, 3)}", f"AUC Mean = {round(auc_mean, 3)}", f"AUC Std. Dev = {round(std_dev, 3)}", f"AUC CI = {confidence_interval}"
