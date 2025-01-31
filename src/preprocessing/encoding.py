
import pandas as pd
import numpy as np

def do_encoding(df):
    #df = clean_01(df)
    df = convert_numerical_to_categorical(df)
    df = target_encoding(df)
    df = log_transform_columns(df, ["count_floors_pre_eq", "age", "area_percentage", "height_percentage", "count_families"])
    df = min_max_scaling(df, ['damage_grade', 'building_id'])
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
        df_encoded[f"{col} t-enc freq/mean"] = df.groupby(col)[target_col].transform('count') / df_encoded[f"{col} t-enc mean"]

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
        df_local[f"{new_col_name} freq/mean"] = combination_key.map(df_local.groupby(combination_key)[target_col].transform('count') / df_local[f"{new_col_name} mean"])

        return df_local.drop(columns=cols)

    # Apply encoding to binary feature combinations
    df_encoded = encode_combination(df_encoded, 'has_superstructure', target_col, 'superstructure t-enc')
    df_encoded = encode_combination(df_encoded, 'has_secondary_use', target_col, 'secondary_usage t-enc')

    return df_encoded

def clean_01(df):
  df = df[df["height_percentage"] <= 18 ] # Drop rows where ‘height_percentage’ is above 18
  return df

def min_max_scaling(df, exclude_columns=None):
    """
    Scales all numerical columns of a DataFrame to the range [0, 1],
    excluding specified columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    exclude_columns (list, optional): List of column names to exclude from scaling.

    Returns:
    pd.DataFrame: Scaled DataFrame.
    """
    df_scaled = df.copy()
    
    # Ensure exclude_columns is a list
    exclude_columns = exclude_columns or []
    
    # Select numerical columns that are not in the exclude list
    numeric_cols = df_scaled.select_dtypes(include=[np.number]).columns
    cols_to_scale = [col for col in numeric_cols if col not in exclude_columns]

    # Apply min-max scaling
    for col in cols_to_scale:
        min_val = df_scaled[col].min()
        max_val = df_scaled[col].max()
        if min_val != max_val:  # Avoid division by zero if constant column
            df_scaled[col] = (df_scaled[col] - min_val) / (max_val - min_val)
        else:
            df_scaled[col] = 0  # If all values are the same, set them to 0

    return df_scaled


def log_transform_columns(df, columns):
    """
    Applies a logarithmic transformation (natural log) to the specified columns in a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to apply the log transformation.

    Returns:
    pd.DataFrame: Transformed DataFrame with log-scaled values.
    """
    df_transformed = df.copy()

    for col in columns:
        if col in df_transformed.columns:
            # Ensure all values are positive to avoid log errors
            min_val = df_transformed[col].min()
            if min_val <= 0:
                df_transformed[col] = np.log1p(df_transformed[col] - min_val + 1)  # log(1 + x) transformation
            else:
                df_transformed[col] = np.log(df_transformed[col])

    return df_transformed
