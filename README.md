# Group Anchovies
## How to run the model
1. Create a .env in the root folder
2. Add 2 variables to the .env exactly like shown below. The values should be the absolute path to the location on your computer, where the images you want to test on are located.
```python
IMAGE_DATA_URL_LOCAL = "/Users/yourname/Documents/itu/projects_in_ds/final-assignment/images"
MASK_DATA_URL_LOCAL = "/Users/yourname/Documents/itu/projects_in_ds/final-assignment/masks"
```
3. If you don't have any premade annotated masks, like we had for training the models, please still create the MASK_DATA_URL_LOCAL variable and set it equal to None.
4. Make sure the images in your folder have the following naming convention:
    - Image names start with 'PAT_' followed by the img_id. E.g. PAT_9_17_80.png.
    - Maks names (if you have any) start with 'PAT_' followed by the img_id + _mask. E.g. PAT_9_17_80_mask.png.
4. In order to succesfully test the model you also need to add the labels of your images. 
    - In the 'data' folder replace the 'metadata.csv' file with your label data. It's very important that you do not rename the file. 'metadata.csv' just needs these exact two columns: 'img_id' and 'diagnostic'. Where MEL is melanoma.
    - Example:

      | img_id       | diagnostic |
      |--------------|------------|
      | 9_17_80      | MEL        |
      | 2_03_45      | NEV        |
      | 7_12_33      | MEL        |
      | 4_22_11      | BKL        |

### Group members:
- Asta Asta Trier Wang
- Bruno Alessandro Damian Modica Figueira
- Jan Peter Cardell
- Maja Kalina Oska
- Philip Münster-Hansen