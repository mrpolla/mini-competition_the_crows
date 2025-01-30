
import pandas as pd

# # 2nd try encoding categorical_columns: Onehot encoding 
def onehot_encode_features(df):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    encoded_df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    assert isinstance(encoded_df, pd.DataFrame)  
    return encoded_df