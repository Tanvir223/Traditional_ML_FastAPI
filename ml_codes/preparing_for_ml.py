import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from ml_codes.helper_functions import *
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error


#----------------------- Label Encoding ----------------------------------
def label_encoder(df):
  date_col, checked_df = find_date_col(df)
  # Identify categorical columns
  categorical_columns = checked_df.select_dtypes(include=['object'])
  label_encoder = LabelEncoder()
  for cat_col in categorical_columns.columns:
    label_encoder.fit(categorical_columns[cat_col])
    with open('saved_files/' + cat_col+'_label_encoder.pkl', 'wb') as f:
      pickle.dump(label_encoder, f)
    checked_df[cat_col] = label_encoder.transform(categorical_columns[cat_col])

  return checked_df, label_encoder

def label_decoder(encoded_df, categorical_cols):

  for cat_col in encoded_df[categorical_cols].columns:
    with open('saved_files/' + cat_col+'_label_encoder.pkl', 'rb') as f:
      loaded_label_encoder = pickle.load(f)
    encoded_df[cat_col] = loaded_label_encoder.inverse_transform(encoded_df[cat_col])
  return encoded_df

def loaded_label_encoder(encoded_df, categorical_cols):
  for cat_col in categorical_cols:
    with open('saved_files/' + cat_col+'_label_encoder.pkl', 'rb') as f:
      loaded_label_encoder = pickle.load(f)
    encoded_df.loc[:,cat_col] = loaded_label_encoder.transform(encoded_df[cat_col])

  return encoded_df

# -------------------------------------------------------------

#-------------------------Scaling -----------------------------
def scaling_encoder(df, numeric_cols):
  scaler = MinMaxScaler()
  for num_col in numeric_cols:
    model=scaler.fit(df[[num_col]])
    with open('saved_files/' + num_col+'_scaling_encoder.pkl', 'wb') as f:
      pickle.dump(model, f)
    df.loc[:,num_col] = model.transform(df[[num_col]])
  return df


def scaling_with_loaded_encoder(df, numeric_cols):
  for num_col in numeric_cols:
    with open('saved_files/' + num_col+'_scaling_encoder.pkl', 'rb') as f:
      scaling_loaded_encoder = pickle.load(f)
    df.loc[:,num_col] = scaling_loaded_encoder.transform(df[[num_col]])
  return df

#--------------------------------------------------------------------

# -----------------------------variable importance --------------------
def get_variable_importance(model, X, y, method='auto', n_repeats=30, random_state=42):
    if method == 'auto':
        if hasattr(model, "feature_importances_"):
            method = 'tree'
        else:
            method = 'permutation'

    if method == 'tree':
        if isinstance(model, (RandomForestRegressor, RandomForestClassifier, XGBRegressor,)):
            return model.feature_importances_
        else:
            raise ValueError("Tree-based feature importance is only available for specific models (Random Forest, XGBoost, LightGBM).")

    if method == 'permutation':
        result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)
        return result.importances_mean

    raise ValueError("Invalid method for variable importance. Use 'tree' or 'permutation'.")

#-------------------------------------------------------------------
#--------------------------------- accuracy metric -------------------


def calculate_accuracy_matric_mae(test_data, test_target, model):

    test_predictions = model.predict(test_data)
    # Calculate MAE testing data

    test_mae = mean_absolute_error(test_target, test_predictions)

    return test_mae