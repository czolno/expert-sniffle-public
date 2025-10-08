import xgboost as xgb
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'risk_estimator'))
from data_loader import get_data

# Prepare your data
to_be_forecasted = 'm_realized_vol_5min_future'

print("Loading training data...")
X_train, y_train = get_data('train')  # Load the training data

print("Loading validation data...")
X_val, y_val = get_data('val')  # Load the validation data

print("Loading test data...")
X_test, y_test = get_data('test')  # Load the test data

# # Create and train the regressor with eval metric = 'rmse'
# model = xgb.XGBRegressor(eval_metric='rmse', verbosity=2)
# model.fit(
#     X_train, y_train,
#     eval_set=[(X_val, y_val)],
#     verbose=True
# )
# model.save_model('xgb_model_rv5min.json')

# # Predict and evaluate
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print("Test MSE:", mse)

# Create and train the regressor with eval metric = 'mae'
model = xgb.XGBRegressor(eval_metric='mae', verbosity=2)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=True
)
model.save_model('xgb_model_rv5min_MAE.json')

# Predict and evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Test MSE:", mse)