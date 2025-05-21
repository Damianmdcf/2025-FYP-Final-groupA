import sys                                     
import pandas as pd                            
from sklearn.tree import DecisionTreeClassifier, export_text  

def decision_tree(filepath, feature_version):
    if filepath.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(filepath)                   # Excel case
    else:
        df = pd.read_csv(filepath)                     # CSV case

    # pick feature columns                          
    feat_cols = feat_cols = [
    "feature A " + feature_version,   # → "feature A (v1)"
    "feature B " + feature_version,   # → "feature B (v1)"
    "feature C " + feature_version]   # → "feature C (v1)"
    X = df[feat_cols]                              # feature matrix
    y = df["Label: Is cancer"]                    # target labels

    # train decision tree                           
    tree_clf = DecisionTreeClassifier(random_state=0)   # reproducible tree
    tree_clf.fit(X, y)                                  # fit on full data

    # show the learnt rules                         
    print(export_text(tree_clf, feature_names=feat_cols))

decision_tree(fr"util\test1.csv", "(v1)")


