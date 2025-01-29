import pandas as pd
from pre_processing.load_data import load_data
from pre_processing.encoding import encode_data
from modeling.train import create_model
from modeling.predict import predict
from data_handler.data_handler import write_CSV

#main pipeline
def run_pipeline():

  #1. Load CSV and Format
  df = load_data()
  assert isinstance(df, pd.DataFrame)  

  #2. Encoding
  df = encode_data(df)
  assert isinstance(df, pd.DataFrame)  

  #3. Training
  model = create_model(df)
  #assert is Model?
  
  #4. Predict
  pred = predict(model, df)
  assert isinstance(pred, pd.DataFrame)  

  #5. Write CSV
  outputh_path = ""
  write_CSV(pred, outputh_path)

  print("Script finished")

if __name__ == "__main__":
  run_pipeline()  
