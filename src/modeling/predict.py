import xgboost as xgb

def predict(model, df): 
  df_test = df[df['train']==0]
#  y_test = df_test['damage_grade']
#  print(y_test)
  
  X_test = df_test.drop('damage_grade', axis=1)

  dtest = xgb.DMatrix(X_test)

  y_pred = model.predict(dtest)
  
  return y_pred