import pandas as pd
from sklearn.model_selection import train_test_split

def data_split(input_csv):
    """
    Split up the data into 80% training data and 20% test data.
    Split is stratified to account for class imbalance.
    """
    # Read the csv file
    df = pd.read_csv(input_csv)
    # Do the stratified split
    train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['Melanoma'],
    random_state=42
    )
    # Write the splits to the data folder
    train_df.to_csv("../data/train-extended-data.csv", index=False)
    test_df.to_csv("../data/test-extended-data.csv", index=False)

if __name__ == "__name__":
    data_split("../data/extended-data-for-model.csv")