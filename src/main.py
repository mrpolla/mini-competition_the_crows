import pandas as pd
from preprocessing.encoding import onehot_encode_features
from modeling.train import create_model
from modeling.predict import predict
from data_handler.data_handler import write_data, load_data
import os

#main pipeline
def run_pipeline():

  os.chdir(os.path.dirname(os.path.abspath(__file__)))
  print(os.getcwd())

  #1. Load CSV and Format
  print("Starting load_data...")
  df = load_data()
  print("load_data done!")

  #2. Encoding
  print("Starting onehot_encode_features...")
  df = onehot_encode_features(df)
  print("onehot_encode_features done!")

  #3. Training
  print("Starting create_model...")
  model = create_model(df)
  print("create_model done!")
  
  #4. Predict
  print("Starting predict...")
  df_pred = predict(model, df)
  print("predict done!")

  #5. Write CSV
  print("Starting write_CSV...")
  write_data(df_pred)
  print("write_CSV done!")

  print("Script finished")

if __name__ == "__main__":
  run_pipeline()  
