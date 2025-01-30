
# Convert suspicious numerical columns to categorical (by converting integers to strings)
# here values are assigned inside definging the fucniton 

def convert_numerical_to_categorical(df):
    numerical_columns_to_convert = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id']
    for col in numerical_columns_to_convert:
        df[col] = df[col].astype(str)  # Convert to string (categorical)
    return df


import pandas as pd

# # 2nd try encoding categorical_columns: Onehot encoding 
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
        # Compute mean target for each category
        mean_target_if_present = df.groupby(col)[target_col].transform('mean')
        # Apply encoding
        df_encoded[col] = mean_target_if_present.values.flatten()
    
    material_cols = df_encoded.filter(like='has_superstructure_').columns
    df_encoded['material_combination'] = df_encoded[material_cols].astype(str).agg(''.join, axis=1)
    mean_target_per_combination = df_encoded.groupby('material_combination')[target_col].transform('mean')
    df_encoded['structure_encoded'] = mean_target_per_combination

    usage_cols = df_encoded.filter(like='has_secondary_use_').columns
    df_encoded['usage_combination'] = df_encoded[usage_cols].astype(str).agg(''.join, axis=1)
    mean_target_per_combination = df_encoded.groupby('usage_combination')[target_col].transform('mean')
    df_encoded['usage_encoded'] = mean_target_per_combination
    
    df_encoded = df_encoded.drop(columns=['material_combination','usage_combination']+list(material_cols)+list(usage_cols))
        
    return df_encoded
