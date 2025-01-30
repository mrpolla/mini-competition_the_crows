import pandas as pd
from preprocessing.encoding import drop_categorical_features
from preprocessing.load_data import load_data
from modeling.train import create_model
from modeling.predict import predict
from data_handler.data_handler import write_CSV

#main pipeline
def run_pipeline():

  #1. Load CSV and Format
  print("Starting load_data...")
  df = load_data()
  assert isinstance(df, pd.DataFrame)  
  print("load_data done!")

  #2. Encoding
  print("Starting drop_categorical_features...")
  df = drop_categorical_features(df)
  assert isinstance(df, pd.DataFrame)  
  print("drop_categorical_features done!")

  #3. Training
  print("Starting create_model...")
  model = create_model(df)
  #assert is Model?
  print("create_model done!")
  
  #4. Predict
  print("Starting predict...")
  pred = predict(model, df)
  pred_df = pd.DataFrame(pred, columns=['damage_grade'])
  assert isinstance(pred_df, pd.DataFrame)  
  print("predict done!")

  #5. Write CSV
  print("Starting write_CSV...")
  output_path = "pred.csv"
  write_CSV(pred_df, output_path)
  print("write_CSV done!")

  print("Script finished")

if __name__ == "__main__":
  run_pipeline()  
