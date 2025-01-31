
import pandas as pd

def do_encoding(df):
#    df = clean_01(df)
    df = convert_numerical_to_categorical(df)
    df = target_encoding(df)
    return df

def convert_numerical_to_categorical(df):
    numerical_columns_to_convert = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
    for col in numerical_columns_to_convert:
        df[col] = df[col].astype(str)  # Convert to string (categorical)
    return df

def onehot_encode_features(df):
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    encoded_df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    assert isinstance(encoded_df, pd.DataFrame)  
    return encoded_df

def target_encoding(df):
    target_col = 'damage_grade'
    df_encoded = df.copy()

    # Encode categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        df_encoded[f"{col} t-enc mean"] = df.groupby(col)[target_col].transform('mean')
        df_encoded[f"{col} t-enc std"] = df.groupby(col)[target_col].transform('std')

    df_encoded.drop(columns=categorical_cols, inplace=True, errors='ignore')  # Drop safely

    # Function for encoding binary combinations
    def encode_combination(df, col_prefix, target_col, new_col_name):
        df_local = df.copy()
        cols = df_local.filter(like=col_prefix).columns
        
        if len(cols) == 0:  # If no matching columns, return original dataframe
            return df_local

        combination_key = df_local[cols].astype(str).agg(''.join, axis=1)
        df_local[f"{new_col_name} mean"] = combination_key.map(df_local.groupby(combination_key)[target_col].transform('mean'))
        df_local[f"{new_col_name} std"] = combination_key.map(df_local.groupby(combination_key)[target_col].transform('std'))

        return df_local.drop(columns=cols)

    df_encoded = encode_combination(df_encoded, 'has_superstructure', target_col, 'superstructure t-enc')
    df_encoded = encode_combination(df_encoded, 'has_secondary_use', target_col, 'secondary_usage t-enc')

    return df_encoded

def clean_01(df):
  df = df[(df["height_percentage"] <= 18) & (df["train"] == 1)] # Drop rows from training set where ‘height_percentage’ is above 18
  return df