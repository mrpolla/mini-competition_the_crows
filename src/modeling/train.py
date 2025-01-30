import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def create_model(df):
  #1. get train data set
  df_train = df[df['train']==1]
  y_train = df_train['damage_grade']
  y_train = y_train - 1 
  print(y_train)

  X_train = df_train.drop('damage_grade', axis=1)

  # Convert data into DMatrix (XGBoost's optimized data structure)
  dtrain = xgb.DMatrix(X_train, label=y_train)

  # Set XGBoost parameters
  params = {
      'objective': 'multi:softmax',  # Use 'multi:softprob' for probabilities
      'num_class': 3,  # Number of unique classes (1, 2, 3)
      'eval_metric': 'mlogloss',
      'max_depth': 3,
      'eta': 0.1
  }

  # Train the model
  num_round = 100  # Number of boosting rounds (iterations)
  bst = xgb.train(params, dtrain, num_round)

  return bst
