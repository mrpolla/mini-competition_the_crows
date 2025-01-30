
# Convert suspicious numerical columns to categorical (by converting integers to strings)
# here values are assigned inside definging the fucniton 

def convert_numerical_to_categorical(df):
    numerical_columns_to_convert = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
    for col in numerical_columns_to_convert:
        df[col] = df[col].astype(str)  # Convert to string (categorical)
    return df

df = convert_numerical_to_categorical(df)




import pandas as pd

# # 2nd try encoding categorical_columns: Onehot encoding 
def onehot_encode_features(df):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    encoded_df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    assert isinstance(encoded_df, pd.DataFrame)  
    return encoded_df