import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def create_model(df):
  #1. get train data set
  df_train = df[df['train']==1]
  temp_y_train = df_train['damage_grade']
  temp_y_train = temp_y_train - 1 
  print(temp_y_train)
  temp_X_train = df_train.drop('damage_grade', axis=1)

  X_train, X_test, y_train, y_test = train_test_split(temp_X_train, temp_y_train, test_size=0.2, random_state=42)

  # Set XGBoost parameters for hypertuning
  param_grid = {
    'n_estimators': [100, 200, 300],        # Number of boosting rounds
    'learning_rate': [0.01, 0.05, 0.1, 0.2], # Step size shrinkage
    'max_depth': [3, 5, 7],                  # Maximum depth of trees
    'min_child_weight': [1, 3, 5],           # Minimum sum of instance weight in a child
    'subsample': [0.7, 0.8, 1.0],            # Fraction of samples used per tree
    'colsample_bytree': [0.7, 0.8, 1.0],     # Fraction of features used per tree
    'gamma': [0, 0.1, 0.2, 0.3],             # Minimum loss reduction to split
    'reg_alpha': [0, 0.01, 0.1],             # L1 regularization
    'reg_lambda': [0.1, 1, 10],              # L2 regularization
  }

  # Define XGBoost model for multi-class classification
  xgb = XGBClassifier(objective='multi:softmax', num_class=len(set(y_train)))
  
  # Perform Grid Search
  grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
  grid_search.fit(X_train, y_train)
  
  # Best hyperparameters
  print("Best Parameters:", grid_search.best_params_)
  
  # Evaluate the best model
  best_model = grid_search.best_estimator_
  print("Best Model Accuracy:", best_model.score(X_test, y_test))

  return best_model

'''
  # Convert data into DMatrix (XGBoost's optimized data structure)
  dtrain = xgb.DMatrix(X_train, label=y_train)
    
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
'''
