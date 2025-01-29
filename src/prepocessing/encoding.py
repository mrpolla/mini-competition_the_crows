

def drop_categorical_features(df):

    # Detect categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Drop categorical columns
    return df.drop(columns=categorical_cols)