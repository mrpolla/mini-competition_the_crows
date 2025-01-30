import xgboost as xgb
import pandas as pd

def predict(model, df): 
  df_test = df[df['train']==0]
  
  X_test = df_test.drop('damage_grade', axis=1)

  dtest = xgb.DMatrix(X_test)

  y_pred = model.predict(dtest)

  df_pred = pd.DataFrame({
    'building_id': df_test['building_id'],
    'damage_grade': y_pred
  })
  
  assert isinstance(df_pred, pd.DataFrame)
  
  return df_pred