import pandas as pd
from sklearn.model_selection import train_test_split

def data_split(input_csv, ):
    df = pd.read_csv(input_csv, )

    train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['diagnostic'],
    random_state=42
    )

    cancerous_types = ['BCC', 'SCC', 'MEL']
    test_df['cancerous'] = test_df['diagnostic'].isin(cancerous_types).astype(int)
    train_df['cancerous'] = train_df['diagnostic'].isin(cancerous_types).astype(int) 
    
    train_df.to_csv("train-model-data.csv", index=False)
    train_df.to_csv("test-model-data.csv", index=False)