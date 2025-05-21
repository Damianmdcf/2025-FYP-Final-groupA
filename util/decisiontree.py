import sys                                     
import pandas as pd                            
from sklearn.tree import DecisionTreeClassifier, export_text  

def decision_tree(filepath, feature_version):
    if filepath.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(filepath)                   
    else:
        df = pd.read_csv(filepath)                     

    # pick feature columns                          
    feat_cols = feat_cols = [
    "feature A " + feature_version,   # → "feature A (v1)"
    "feature B " + feature_version,   # → "feature B (v1)"
    "feature C " + feature_version]   # → "feature C (v1)"
    X = df[feat_cols]                              
    y = df["Label: Is cancer"]                    

    # Train my decision tree                           
    tree_clf = DecisionTreeClassifier(random_state=0)   
    tree_clf.fit(X, y)                                  

    # Show the learning groups                         
    print(export_text(tree_clf, feature_names=feat_cols))  

decision_tree(fr"util\test1.csv", "(v1)")


