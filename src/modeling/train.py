import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import optuna
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from functools import partial

def create_model_lightgbm(df):
  #1. get train data set
  df_train = df[df['train']==1]
  temp_y_train = df_train['damage_grade']
  temp_y_train = temp_y_train - 1 
  print(temp_y_train)
  temp_X_train = df_train.drop('damage_grade', axis=1)

  X_train, X_test, y_train, y_test = train_test_split(temp_X_train, temp_y_train, test_size=0.2, random_state=42)

## Hyperparameter tuning
  # Run Optuna Optimization
  objective_with_data = partial(objective, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
  study = optuna.create_study(direction="maximize")
  study.optimize(objective_with_data, n_trials=50)
  
  # Best Parameters
  print("Best Hyperparameters:", study.best_params)
  
  # Train the final model with best params
  best_params = study.best_params
  best_model = lgb.LGBMClassifier(**best_params, random_state=42)
  best_model.fit(X_train, y_train)
  
  # Evaluate Final Model
  y_pred = best_model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  print("Best Model Accuracy:", accuracy)
  best_model.save_model('lgbm_model.txt')
##

  return best_model


def objective(trial, X_train, X_test, y_train, y_test):
    params = {
        "n_estimators": 50,  # Fixed for simplicity, you can tune this too if needed
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),  # Tuning learning rate
        "num_leaves": trial.suggest_int("num_leaves", 20, 150, step=10),  # Tuning number of leaves
        "max_depth": trial.suggest_int("max_depth", 3, 10),  # Tuning max depth
        "min_child_samples": 20,  # Fixed for simplicity, you can tune if needed
        "subsample": 0.8,  # Fixed for simplicity
        "colsample_bytree": 0.8  # Fixed for simplicity
    }
    
    # Train LightGBM model
    model = lgb.LGBMClassifier(**params, random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="logloss", callbacks=[lgb.early_stopping(stopping_rounds=50), log_callback])
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    return accuracy_score(y_test, y_pred)

def log_callback(env):
    print(f"Iteration {env.iteration}, Train metric: {env.evaluation_result_list[0][2]}")

def create_model_with_CV(df):
  #1. get train data set
  df_train = df[df['train']==1]
  temp_y_train = df_train['damage_grade']
  temp_y_train = temp_y_train - 1 
  temp_X_train = df_train.drop('damage_grade', axis=1)

  X_train, X_test, y_train, y_test = train_test_split(temp_X_train, temp_y_train, test_size=0.2, random_state=42)

  # Set XGBoost parameters for hypertuning
  param_grid = {
    'n_estimators': [100, 200, 300],         # Number of boosting rounds
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
  xgb = xgb.XGBClassifier(objective='multi:softmax', num_class=len(set(y_train)))
  
  # Perform Grid Search
  grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
  grid_search.fit(X_train, y_train)
  
  # Best hyperparameters
  print("Best Parameters:", grid_search.best_params_)
  
  # Evaluate the best model
  best_model = grid_search.best_estimator_
  print("Best Model Accuracy:", best_model.score(X_test, y_test))

  return best_model

def create_model_simple(df):
  #1. get train data set
  df_train = df[df['train']==1]
  y_train = df_train['damage_grade']
  y_train = y_train - 1 

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
