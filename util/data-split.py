import pandas as pd
from sklearn.model_selection import train_test_split

def data_split(input_csv):
    df = pd.read_csv(input_csv)

    train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['Melanoma'],
    random_state=42
    )

    train_df.to_csv("../data/train-baseline-data.csv", index=False)
    test_df.to_csv("../data/test-baseline-data.csv", index=False)

data_split("../data/baseline-data-for-model.csv")