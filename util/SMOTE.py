import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
import matplotlib.pyplot as plt

#Resampling of the data, should be perfomered after data augmentation as we have very little melanoma cases
#I created two functions, one uses only SMOT# [creating synthethic samples by interpolating between real minority samples] 
#and the other applies both SMOTE on minority class and undersampling on majority class, have to decide which one works better


df_abc= pd.read_csv("../data/train-baseline-data.csv")

def clean_abc_data(df, feature_cols):
    #removing rows with missing values in any of the feature columns
    df_clean = df.dropna(subset=feature_cols)
    return df_clean


df_clean= clean_abc_data(df_abc, ['feature_a', 'feature_b', 'feature_c'])



def apply_smote_only(df, feature_cols, id_col, label_col, sampling_ratio=0.5, k_neighbors=5):
    #Undersamples the minority class using SMOTE [creating synthethic samples by interpolating between real minority samples]
    X = df[feature_cols].values
    y = df[label_col].values
    ids = df[id_col].values

    #appplying SMOTE, could be good to try different k_neighbours
    #sampling strategy= oversample the minority class to half the size of the majority one
    smote = SMOTE(sampling_strategy=sampling_ratio, k_neighbors=k_neighbors, random_state=42)
    X_s, y_s = smote.fit_resample(X, y)

    #print the amount of samples of each class
    print("After SMOTE:", Counter(y_s))

    #created a data frame with added samples
    df_s = pd.DataFrame(X_s, columns=feature_cols)
    df_s[label_col] = y_s
    df_s[id_col] = ['synthetic_' + str(i) if i >= len(ids) else ids[i] for i in range(len(df_s))]

    return df_s


def apply_smote_and_undersample(df, feature_cols, id_col, label_col,
                                 smote_ratio=0.3, under_ratio=0.5, k_neighbors=5):

    X = df[feature_cols].values
    y = df[label_col].values
    ids = df[id_col].values
    #oversample the minority class using SMOTE, adds synenthis samples, #0.1, try to change k_values
    over = SMOTE(sampling_strategy=smote_ratio, k_neighbors=k_neighbors, random_state=42)
    #undersample the majority class, ranodmly removes non-melanoma cases, reduces the risk of overfitting to the overrepresented majority class
    under = RandomUnderSampler(sampling_strategy=under_ratio, random_state=42)
    #build pipeline to first apply SMOTE then undersampling
    pipeline = Pipeline(steps=[('o', over), ('u', under)])
    X_ou, y_ou= pipeline.fit_resample(X, y)

    df_ou = pd.DataFrame(X_ou, columns=feature_cols)
    df_ou[label_col] = y_ou
    df_ou[id_col] = ['synthetic_' + str(i) if i >= len(ids) else ids[i] for i in range(len(df_ou))]

    #print the amount of samples after performing SMOTE and undersampling
    print("After SMOTE + u:", Counter(y_ou))

    return df_ou



#Apply smote to trained data
df_smote= apply_smote_only(df_clean, ['feature_a', 'feature_b', 'feature_c'], "img_id", "Melanoma")
#save the data framework
df_smote.to_csv("../data/abc_features_smote.csv", index= False)



#Apply smote and under_sampling
df_smote_u= apply_smote_and_undersample(df_clean, ['feature_a', 'feature_b', 'feature_c'], "img_id", "Melanoma")
df_smote_u.to_csv("../data/abc_features_ou.csv", index= False)
