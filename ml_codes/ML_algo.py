from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from ml_codes.preparing_for_ml import get_variable_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


class RegressionModels:
  def __init__(self):
    print("---ML_Models class defined")
  def model_LR(self, X, y, stage='training', best_params={}):
      if stage=='training':
        # Define the parameter grid for Grid Search
        param_grid = {
            'fit_intercept': [True, False]
        }

        # Create a Linear Regression model
        lr = LinearRegression()
        print("Training with Linear Regression . . .")
        # Create a Grid Search Cross-Validation object with MAE as the scoring metric
        grid_search = GridSearchCV(estimator=lr, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=3)

        # Fit the model to the data and find the best hyperparameters
        grid_search.fit(X, y)

        # Get the best hyperparameters
        best_params = grid_search.best_params_

      # Train the model with the best hyperparameters
      best_lr_model = LinearRegression(fit_intercept=best_params['fit_intercept'])
      best_lr_model.fit(X, y)

      var_imp = get_variable_importance(best_lr_model, X, y, method='auto', n_repeats=30, random_state=42)

      # Calculate cross-validation MAE
      cross_val_mae = cross_val_score(best_lr_model, X, y, scoring='neg_mean_absolute_error', cv=3)
      cross_val_mae = -cross_val_mae  # Convert to positive MAE values
      print("Complete Training with Linear Regression . . .")
      return best_lr_model, best_params, cross_val_mae, var_imp

  def model_XGB(self, X, y, stage='training', best_params={}):
      if stage=='training':
        # Define the parameter grid for Grid Search
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.01, 0.1, 0.2]
        }
        print("Training with Xtreme Gradient Boosting . . .")
        # Create an XGBoost Regressor
        xgb = XGBRegressor()

        # Create a Grid Search Cross-Validation object with MAE as the scoring metric
        grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=3)

        # Fit the model to the data and find the best hyperparameters
        grid_search.fit(X, y)

        # Get the best hyperparameters
        best_params = grid_search.best_params_

      # Train the model with the best hyperparameters
      best_xgb_model = XGBRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], learning_rate=best_params['learning_rate'])
      best_xgb_model.fit(X, y)
      var_imp = get_variable_importance(best_xgb_model, X, y, method='auto', n_repeats=30, random_state=42)

      # Calculate cross-validation MAE
      cross_val_mae = cross_val_score(best_xgb_model, X, y, scoring='neg_mean_absolute_error', cv=3)
      cross_val_mae = -cross_val_mae  # Convert to positive MAE values
      print("Complete Training with Xtreme Gradient Boosting . . .")
      return best_xgb_model, best_params, cross_val_mae, var_imp

  def model_RF(self, X, y, stage='training', best_params={}):
    if stage == 'training':
        # Define the parameter grid for Grid Search
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        print("Training with Random Forest . . .")
        # Create a Random Forest Regressor
        rf = RandomForestRegressor()

        # Create a Grid Search Cross-Validation object with MAE as the scoring metric
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=3)

        # Fit the model to the data and find the best hyperparameters
        grid_search.fit(X, y)

        # Get the best hyperparameters
        best_params = grid_search.best_params_

    # Train the model with the best hyperparameters
    best_rf_model = RandomForestRegressor(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf']
    )
    best_rf_model.fit(X, y)
    var_imp = get_variable_importance(best_rf_model, X, y, method='auto', n_repeats=30, random_state=42)
    # Calculate cross-validation MAE
    cross_val_mae = -cross_val_score(best_rf_model, X, y, scoring='neg_mean_absolute_error', cv=3)

    print("Complete Training with Random Forest . . .")
    return best_rf_model, best_params, cross_val_mae, var_imp




