import pandas as pd

def write_data(df):
    
    try:
        df.to_csv('../data/processed/train_values.csv')
    except:
        print('Error. Could not write file.')