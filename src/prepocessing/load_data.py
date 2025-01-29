import pandas as pd

def load_data():

    try:
        df_train_X = pd.read_csv('../data/raw/train_values.csv')
        df_train_y = pd.read_csv('../data/raw/train_labels.csv')
        df_test_X = pd.read_csv('../data/raw/test_values.csv')
    except FileNotFoundError:
        print("Error: File not found. Please check the file path.")

    nrows_train = df_train_X.shape[0]
    nrows_test = df_test_X.shape[0]
    nrows_tot = nrows_train+nrows_test

    df_train_X['train'] = 1
    df_test_X['train'] = 0
    df_X = pd.concat([df_train_X, df_test_X], ignore_index=True)
    df = pd.merge(df_X, df_train_y, on='building_id', how='left')

    assert df.shape[0]==nrows_tot, "Incorrect number of rows"
    assert isinstance(df, pd.DataFrame), "No pandas dataframe returned"

    return df
