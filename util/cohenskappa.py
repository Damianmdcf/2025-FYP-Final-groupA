import pandas as pd
from sklearn.metrics import cohen_kappa_score
from scipy.stats import mode

# Cohens Kappa function to compare Human Annotations with the output of the Hair-Removal Function
def Cohenskappa(annotations, function_output):

    # Use Majorty Vote for our Annotations to get Categorical Data (used as Ground-Truth)
    majority_labels = mode(annotations.values, axis=1, keepdims=False).mode

    # calculate the Cohens Kappa
    kappa = cohen_kappa_score(majority_labels, function_output)

    return kappa
