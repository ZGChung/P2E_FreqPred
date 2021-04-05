import pandas as pd
import numpy as np
from datetime import datetime
import time
import sklearn.metrics as metrics

print("Hello world! Program begins.")
df = pd.read_csv("data_daily.csv")
'''
# test read file
print(df.shape)
print(df.info())
df.head(5)
'''
# convert the first column to datetime format
df['time'] = pd.to_datetime(df['time'], unit='s')
df = df.set_index('time')
print(df)

# Use the data of 2019 as training set
# Use the data pre-COVID of 2020 as testing set

# Helper functions, the metrics for evaluations
'''
def regression_results(y_true, y_pred):
    # Regression metrics
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred)
    mse=metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error=metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    print('explained_variance: ', round(explained_variance,4))
    print('mean_squared_log_error: ', round(mean_squared_log_error,4))
    print('r2: ', round(r2,4))
    print('MAE: ', round(mean_absolute_error,4))
    print('MSE: ', round(mse,4))
    print('RMSE: ', round(np.sqrt(mse),4))
'''
