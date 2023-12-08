from ml_codes.helper_functions import *
from ml_codes.preparing_for_ml import *
from ml_codes.ML_algo import RegressionModels
from helper.mongodb_helper_functions import *
import random

def reg_feature_engineering(df, selected_features, target_column, select_ml_algo):

    cleaned_df = df.copy()
    cleaned_df = cleaned_df.dropna()

    #-------------------------------
    feature_columns =  cleaned_df.drop(columns= [target_column]).columns
    feature_columns = list(set(feature_columns).intersection(set(selected_features)))

    X = cleaned_df[feature_columns]
    y = cleaned_df[target_column]
     
    # extract numeric and categorical type column name
    date_col, x_df = find_date_col(X)
    # Identify categorical columns
    categorical_columns = x_df.select_dtypes(include=['object'])
    categorical_columns = list(categorical_columns.columns)
    
    # Identify numeric columns
    numeric_columns = x_df.select_dtypes(include=['number'])
    numeric_columns = list(numeric_columns.columns)

    test_size = 0.2

    # Split the data into training and testing sets
    random.seed(1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    df_train = pd.concat([X_train, y_train],axis=1)
    print("---train dataset shape = ",df_train.shape)
    df_test = pd.concat([X_test, y_test],axis=1)
    print("---test dataset shape = ",df_test.shape)

    #---- label encoding--------
    encoded_df_train, _ = label_encoder(df_train)
    encoded_df_test = loaded_label_encoder(df_test, categorical_columns)

    #---- scaling--------
    encoded_df_train = scaling_encoder(encoded_df_train, numeric_columns)
    scaled_df_train = encoded_df_train.copy()
    encoded_df_test = scaling_with_loaded_encoder(encoded_df_test, numeric_columns)
    scaled_df_test = encoded_df_test.copy()

    # ---- separate the processed feature  and the target-----
    X_train, X_test, y_train, y_test = scaled_df_train[feature_columns], scaled_df_test[feature_columns], scaled_df_train[target_column], scaled_df_test[target_column]

    # -------- creating ml model object and make it dynamic based on algo names
    validation_result_df = pd.DataFrame(columns = ['algorithm','train_score', 'test_score','best_params'])
    variable_importance_df = pd.DataFrame(columns = ["algorithm"] + list(feature_columns))
    ml_models_obj = RegressionModels()

    for ml_algo in select_ml_algo:
        print("-------ml_algo------------",ml_algo)
        algo_run_function = "model_"+ ml_algo
        extract_function = getattr(ml_models_obj, algo_run_function)
        # print(extract_function())
        fitted_model , best_params, train_score, var_importance = extract_function(X_train, y_train, stage='training')
        test_score = calculate_accuracy_matric_mae(X_test, y_test, fitted_model)
        validation_result_df.loc[len(validation_result_df)] =  [ml_algo, np.mean(train_score), test_score, best_params]
        variable_importance_df.loc[len(variable_importance_df)] = [ml_algo] + list(var_importance)
        
    validation_result_df['guid'] = df['guid'][0]
    variable_importance_df['guid'] = df['guid'][0]
    save_data(validation_result_df, 'evalution_colections')
    save_data(variable_importance_df, 'variable_imp_colections')
    # return validation_result_df, variable_importance_df
