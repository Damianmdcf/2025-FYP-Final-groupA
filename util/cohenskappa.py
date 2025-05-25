
from scipy.stats import mode
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import cv2
import numpy as np
import os
from img_util import readImageFile
from inpaint_util import removeHair


#Converts binary hair mask to a discrete score: 0 = no hair detected (mask all zeros), 1 = some hair (non-zero, but low ratio), 2 = lots of hair (high ratio)
def compute_hair_score(thresh_mask, high_thresh=0.3): 
    #Counts the number of pixels decided to be removed in the remove function
    hair_pixels = np.sum(thresh_mask > 0)
    total_pixels = thresh_mask.size

    #Calculate the proportion of image covered in hair 
    hair_ratio = hair_pixels / total_pixels

    if hair_pixels == 0: #set to zero, as we agreed we say ) if and only if there is not a single hair in the pic
        return 0
    elif hair_ratio < high_thresh: #0.3 value decided looking at the hitsogram of hair ratios, on ipynb file Maja's computer
        return 1
    else:
        return 2
    

# Cohens Kappa function to compare Human Annotations with the output of the Hair-Removal Function
def Cohenskappa(annotations, function_output):

    # Drop 'img_id' column to keep only annotator scores
    annotation_data = annotations.drop(columns=["img_id"])
    # Compute majority vote across annotators
    majority_labels = annotation_data.mode(axis=1)[0].astype(int).values
    
    # calculate the Cohens Kappa
    kappa = cohen_kappa_score(majority_labels, function_output)

    return kappa


folder_path = "data_images" 

#Calculate computer score for the 200 images we annotated 
df_metadata = pd.read_csv("metadata.csv")
image_ids = df_metadata["img_id"][:200] 

hair_scores = []

for image_id in image_ids:
    fpath = os.path.join(folder_path, image_id)
    if not os.path.exists(fpath):
        print(f"Skipping missing file: {fpath}")
        continue
    try:
        img_rgb, img_gray = readImageFile(fpath)
        _, thresh, _ = removeHair(img_rgb, img_gray)
        score = compute_hair_score(thresh)
        hair_scores.append((image_id, score))
    except Exception as e:
        print(f"Error processing {image_id}: {e}")

#Data frame with computer scores
df_hairscores = pd.DataFrame(hair_scores, columns=["img_id", "auto_score"])

# Load annotations
annotations = pd.read_csv("annotations.csv")

# Load model outputs
computer_assesment= df_hairscores["auto_score"].astype(int).values


print(Cohenskappa(annotations, computer_assesment))
#Even tiny shadows, edges, or contrast noise might get picked up as non-zero pixels in thresh.