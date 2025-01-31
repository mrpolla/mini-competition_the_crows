
import pandas as pd

def do_encoding(df):
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
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        mean_target_if_present = df.groupby(col)[target_col].transform('mean')
        df_encoded[f"{col} t-enc"] = mean_target_if_present.values.flatten()
    df_encoded = df_encoded.drop(columns=categorical_cols)
    
    def encode_combination(col_prefix, target_col, new_col_name):
        cols = df_encoded.filter(like=col_prefix).columns
        df_encoded[new_col_name] = df_encoded[cols].astype(str).agg(''.join, axis=1)
        df_encoded[new_col_name] = df_encoded.groupby(new_col_name)[target_col].transform('mean')
        return df_encoded.drop(columns=list(cols))

    df_encoded = encode_combination('has_superstructure_', target_col, 'structure_encoded')
    df_encoded = encode_combination('has_secondary_use', target_col, 'usage_encoded')
        
    return df_encoded
