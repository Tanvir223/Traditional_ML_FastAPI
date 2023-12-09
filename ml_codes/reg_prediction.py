from ml_codes.helper_functions import *
from ml_codes.preparing_for_ml import *
from helper.mongodb_helper_functions import *
import random

def predict(df, select_finalized_variables, select_final_ml_algo):
    # extract numeric and categorical type column name
    date_col, checked_df = find_date_col(df)
    checked_df = checked_df[select_finalized_variables]
    # Identify categorical columns
    categorical_columns = checked_df.select_dtypes(include=['object'])
    categorical_columns = list(categorical_columns.columns)

    # Identify numeric columns
    numeric_columns = checked_df.select_dtypes(include=['number'])
    numeric_columns = list(numeric_columns.columns)

    #---- label encoding--------
    checked_df = loaded_label_encoder(checked_df, categorical_columns)

    #---- scaling--------
    checked_df= scaling_with_loaded_encoder(checked_df, numeric_columns)

    for ml_algo in select_final_ml_algo:
        loaded_model = joblib.load('saved_files/'+ml_algo+'.sav')
        df['predict_with_'+ml_algo] = loaded_model.predict(checked_df)
    save_data(df, 'predicted_production_collections')
    print("----Prediction is completed---")