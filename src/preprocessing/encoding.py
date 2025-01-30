
import pandas as pd

# # 2nd try encoding categorical_columns: Onehot encoding 
def onehot_encode_features(df, categorical_columns):
    return pd.get_dummies(df, columns=categorical_columns, drop_first=True)
# Apply one-hot encoding to the dataframe
encoded_df = onehot_encode_features(df, categorical_columns)
print(encoded_df.columns)
print(f'Total number of columns after encoding: {len(encoded_df.columns)}')


# possible Assert?!:
# for col in categorical_columns:
#         assert col not in encoded_df.columns, f"Error: {col} is still in the dataframe after encoding."



# 1st try encoding categorical_columns: droping 
# def drop_categorical_features(df):
#     # Detect categorical columns
#     categorical_cols = df.select_dtypes(include=['object', 'category']).columns

#     assert isinstance(df, pd.DataFrame)  

#     # Drop categorical columns
#     return df.drop(columns=categorical_cols)
