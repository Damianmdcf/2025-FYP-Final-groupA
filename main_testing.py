import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report, precision_score, recall_score, roc_curve

def main(train_csv, test_csv, result_path, threshold=0.03):
    # Load CSVs for training and testing
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Select the z-score feature columns, since we train on the standardized values
    feat_cols = ["Z_feature_a", "Z_feature_b", "Z_feature_c"]

    # Drop rows with NaN values
    train_df = train_df.dropna(subset=feat_cols + ["Melanoma"])
    test_df = test_df.dropna(subset=feat_cols + ["Melanoma"])

    # Extract features and labels
    x_train = train_df[feat_cols]
    y_train = train_df["Melanoma"]

    x_test = test_df[feat_cols]
    y_test = test_df["Melanoma"]

    # Train the logistic regression model on training data
    clf = LogisticRegression()
    clf.fit(x_train, y_train)

    # Predict probabilities on the test set
    y_prob = clf.predict_proba(x_test)[:, 1]  # Probability for Melanoma

    # Apply threshold to get predicted labels
    y_pred = (y_prob >= threshold).astype(int)

    # Get evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save prediction results to csv
    result_df = test_df[["img_id"]].copy()
    result_df["true_label"] = y_test
    result_df["predicted_label"] = y_pred
    result_df["predicted_proba"] = y_prob
    result_df.to_csv(f"{result_path}_predictions.csv", index=False)

    # Save metrics to csv
    metrics = {
        "Threshold": threshold,
        "Accuracy": acc,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "AUC": auc,
        "TN": tn, "FP": fp, "FN": fn, "TP": tp
    }
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"{result_path}_metrics.csv", index=False)



if __name__ == "__main__":
    train_csv = "data/train-extended-data.csv"
    test_csv = "data/test-extended-data.csv"
    result_path = "result/results_extended"

    main(train_csv, test_csv, result_path)
