import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.stats import norm
from sklearn.metrics import confusion_matrix
import os
from pathlib import Path

droot= Path("data")

def logistic_regression(file_path, treshold):
    df = pd.read_csv(file_path)

    feat_cols = ["Z_feature_a", "Z_feature_b", "Z_feature_c"]
    df= df.dropna(subset=feat_cols + ["Melanoma"]) 


    features = df[feat_cols]
    labels = df["Melanoma"]

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs, f1s, aucs = [], [], []
    fold_rows= []

    for fold, (train_index, test_index) in enumerate(kf.split(features, labels), 1):
        
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]

        model = LogisticRegression()
        model.fit(X_train, y_train)

        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs > float(treshold)).astype(int)

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_probs)

        # save per-fold lists (for the summary)
        accs.append(acc); f1s.append(f1); aucs.append(auc)

        fold_rows.append({
        "Model"      : f"logistic regression {treshold}",
        "Fold"       : fold,
        "Accuracy"   : round(acc, 3),
        "F1"         : round(f1, 3),
        "AUC"        : round(auc, 3)})

    accuracy = np.mean(accs)
    F1 = np.mean(f1s)
    auc_mean = np.mean(aucs)
    std_dev = np.std(aucs)

    z = norm.ppf(0.975)
    margin = z * std_dev / np.sqrt(kf.get_n_splits())
    confidence_interval = (round(auc_mean - margin, 3), round(auc_mean + margin, 3))

    per_fold_path = droot / "result-fold-metrics-extended.csv"

    # wide = pd.DataFrame(fold_rows).pivot(index="Model", columns="Fold", values=["F1", "AUC"])
    # wide.columns = [f"{m} fold {f}" for m, f in wide.columns]      # flatten names
    # wide.reset_index().to_csv(per_fold_path,
    #                        mode="a", index=False,
    #                        header=not os.path.exists(per_fold_path))


    result_df = pd.DataFrame([{
        "Model": f"logistic regression {treshold}",
        "Accuracy Mean": round(accuracy, 3),
        "F1 Mean": round(F1, 3),
        "AUC Mean": round(auc_mean, 3),
        "AUC Std. Dev": round(std_dev, 3),
        "AUC 95% CI": confidence_interval
        }])
    
    # result_csv_path = droot / "result-hair-removal.csv"
    # result_df.to_csv(result_csv_path, mode='a', index=False, header=not os.path.exists(result_csv_path))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return result_df, aucs

file_path= droot / "train-extended-data.csv"
logistic_regression(file_path, 0.01)