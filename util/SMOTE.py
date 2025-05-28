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


df_clean= clean_abc_data(df_abc, ['Z_feature_a', 'Z_feature_b', 'Z_feature_c'])



def apply_smote_only(df, feature_cols, id_col, label_col, sampling_ratio=0.3, k_neighbors=5):
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
    df_s = pd.DataFrame([{
    'img_id': ids[i] if i < len(ids) else f'synthetic_{i - len(ids)}',
    'Melanoma': y_s[i],
    'Z_feature_a': X_s[i][0],
    'Z_feature_b': X_s[i][1],
    'Z_feature_c': X_s[i][2]
    } for i in range(len(X_s))])


    return df_s


def apply_smote_and_undersample(df, feature_cols, id_col, label_col,
                                 smote_ratio=0.3, under_ratio=0.5, k_neighbors=5):
    
    #Extract data
    X = df[feature_cols]
    y = df[label_col]
    ids = df[id_col].to_list()

    #oversample the minority class using SMOTE, adds synenthis samples, #0.1, try to change k_values
    smote = SMOTE(sampling_strategy=smote_ratio, k_neighbors=k_neighbors, random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    # Identify how many synthetic samples were added, neccessary to correctly label data, as this time we are not only adding new data but later randomly deleting cases from majority class
    n_original = len(df)
    n_synthetic = len(X_smote) - n_original

    # Create new IDs for synthetic samples
    new_ids = ids + [f"synthetic_{i}.jpg" for i in range(n_synthetic)]

    # Create new DataFrame 
    df_smote = pd.DataFrame(X_smote, columns=feature_cols)
    df_smote[label_col] = y_smote
    df_smote[id_col] = new_ids

    #undersample the majority class, ranodmly removes non-melanoma cases, reduces the risk of overfitting to the overrepresented majority class
    rus = RandomUnderSampler(sampling_strategy=under_ratio, random_state=42)
    X_final, y_final= rus.fit_resample(X_smote, y_smote)

    #print the amount of samples after performing SMOTE and undersampling
    print("After SMOTE + u:", Counter(y_final))

    final_indices= rus.sample_indices_
    df_final= df_smote.iloc[final_indices].copy()

    # Return the augmentanted dataframe
    return df_final
    



for k in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #Apply smote to trained data
    df_smote= apply_smote_only(df_clean, ['Z_feature_a', 'Z_feature_b', 'Z_feature_c'], "img_id", "Melanoma", sampling_ratio= k)
    #save the data framework
    df_smote.to_csv(f"../data/train-smote-data-{str(k).replace('.','')}.csv", index= False)




for k in [0.1, 0.2, 0.3, 0.4, 0.5]:
    for j in [0.3, 0.5, 0.7]:
    ##Apply smote and under_sampling
        df_smote_u= apply_smote_and_undersample(df_clean, ['Z_feature_a', 'Z_feature_b', 'Z_feature_c'], "img_id", "Melanoma", smote_ratio= k, under_ratio= j)
    #save the data framework
        df_smote_u.to_csv(f"../data/train-smote+under-sampling-data-{str(k).replace('.','')}.csv", index= False)

