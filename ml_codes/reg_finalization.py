from ml_codes.ML_algo import RegressionModels
from ml_codes.helper_functions import *
from ml_codes.preparing_for_ml import *

def finalize_model(X_train, y_train, X_test, y_test, select_ml_algo, select_final_variables, validation_result_df):
  # extract numeric and categorical type column name
  X_train = X_train[select_final_variables].copy()
  X_test = X_test[select_final_variables].copy()

  date_col, x_df = find_date_col(X_train)
  # Identify categorical columns
  categorical_columns = x_df.select_dtypes(include=['object'])
  categorical_columns = list(categorical_columns.columns)
  print("-----",categorical_columns)
  # Identify numeric columns
  numeric_columns = x_df.select_dtypes(include=['number'])
  numeric_columns = list(numeric_columns.columns)

  ml_models_obj = RegressionModels()
  for ml_algo in select_ml_algo:
    algo_run_function = "model_"+ ml_algo
    extract_function = getattr(ml_models_obj, algo_run_function)
    # print(extract_function())
    loaded_best_params = dict(validation_result_df[validation_result_df['algorithm']==ml_algo]['best_params'])
    fitted_model , best_params, train_score, var_importance = extract_function(X_train, y_train, stage='finalize', best_params=loaded_best_params[1])

    joblib.dump(fitted_model, 'saved_models/'+ml_algo+'.sav')

    X_train["predicted_"+str(pd.DataFrame(y_train).columns[0])+ml_algo] = fitted_model.predict(X_train)
    X_train['train_or_test'] = 'train'
    X_test["predicted_"+str(pd.DataFrame(y_test).columns[0])+ml_algo] = fitted_model.predict(X_test)
    X_test['train_or_test'] = 'test'

    predicted_train_df = pd.concat([X_train, y_train], axis=1)
    predicted_test_df = pd.concat([X_test, y_test], axis=1)

    full_predicted_df = pd.concat([predicted_train_df, predicted_test_df])
    full_predicted_df = label_decoder(full_predicted_df, categorical_cols=categorical_columns)

    full_predicted_df.to_csv("predicted_data/predicted_train_test.csv",index=False)



