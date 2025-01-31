import pandas as pd
from preprocessing.encoding import *
from modeling.train import create_model_simple
from modeling.predict import predict
from data_handler.data_handler import write_data, load_data
import os


# main pipeline
def run_pipeline():

  os.chdir(os.path.dirname(os.path.abspath(__file__)))

  #1. Load CSV and Format
  print("Loading data...", end=" ")
  df = load_data()
  print("DONE")

  #2. Encoding
  print("Encoding...", end=" ")
  df = do_encoding(df)
  print("DONE")

  #3. Training
  print("Training model...", end=" ")
  model = create_model_simple(df)
  print("DONE")
  
  #4. Predict
  print("Making predictions...", end=" ")
  df_pred = predict(model, df)
  print("DONE")

  #5. Write CSV
  print("Writing predictions to file...", end=" ")
  write_data(df_pred)
  print("DONE")

  print("Script finished")

if __name__ == "__main__":
  run_pipeline()  
