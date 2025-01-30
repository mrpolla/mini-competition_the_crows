
import pandas as pd

def drop_categorical_features(df):
    # Detect categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    assert isinstance(df, pd.DataFrame)  

    # Drop categorical columns
    return df.drop(columns=categorical_cols)