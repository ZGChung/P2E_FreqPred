from warnings import simplefilter
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
import sklearn as sk
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsRegressor
from sktime.utils.plotting import plot_series
from sktime.forecasting.compose import (
    EnsembleForecaster,
    # MultiplexForecaster,
    ReducedForecaster,
    TransformedTargetForecaster,
)
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    ForecastingGridSearchCV,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.bats import BATS
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.performance_metrics.forecasting import sMAPE, smape_loss
from sktime.transformations.series.detrend import Deseasonalizer, Detrender

simplefilter("ignore", FutureWarning)
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', RuntimeWarning)

NumberOfPredictions = 3

print("Hello world! Program begins.")
df1 = pd.read_csv("data_daily_preCOVID_2cols.csv")

# test read file
print("df1.shape", df1.shape)
print("df1", df1)

df = df1.loc[df1["entries_daily"] != 0]
df = df.reset_index()
print("df.shape", df.shape)
print("df", df)

# convert the first column to datetime format
# df['time'] = pd.to_datetime(df['time'], unit = 's')
# print(df)
# df = df.set_index('time')
y = pd.Series(data = df['entries_daily'])
# x = df.time
# y = df.entries_daily

# Use the data of 2019 as training set, marked in blue in the plot
# Use the data pre-COVID of 2020 as testing set, marked in orange in the plot
# fig1, ax1 = plot_series(y)
# plt.show()
y_train, y_test = temporal_train_test_split(y, test_size = 42)
# fig2, ax2 = plot_series(y_train, y_test, labels = ["y=train", "y=test"])
# ax2.set_title("Original data after Train-Test separation")
# plt.show()
# print(y_train.shape[0], y_test.shape[0])
# use a forecasting horizon the same size as the test set
fh = np.arange(len(y_test)+1)
# print(fh)

'''
# predicting with the last value
# a naive test just to verify the model works
forecaster = NaiveForecaster(strategy = "last")
forecaster.fit(y_train)
y_pred_NaiveForecaster = forecaster.predict(fh)
fig3, ax3 = plot_series(y_train, y_test, y_pred_NaiveForecaster, labels = ["y_train", "y_test", "y_pred"])
ax3.set_title("Naive Forecaster: predict directly the final value")
plt.show()
# we use sMAPE as the evaluation metric here
# sMAPE represents: symmetric Mean Absolute Percentage Error
y_pred_NaiveForecaster = y_pred_NaiveForecaster.drop(y_pred_NaiveForecaster.index[0])
loss3 = smape_loss(y_pred_NaiveForecaster, y_test)
print("The sMAPE for NaiveForecaster method is:", loss3)
'''

# predicting with kNN

# search the k for the kNN minimizing the sMAPE
listOfsMAPE = []

listOfsMAPE.append(20)  # initialize the first as a big number
rangeMax = 324
for i in range(1,rangeMax):
    regressor = KNeighborsRegressor(n_neighbors = i)
    forecaster = ReducedForecaster(
        regressor, scitype = "regressor", window_length = 15, strategy = "recursive"
    )
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    y_pred = y_pred.drop(y_pred.index[0])
    loss = smape_loss(y_test, y_pred)
    print("The sMAPE loss for ", i,"NN prediction is:", loss)
    listOfsMAPE.append(loss)
# search the min of sMAPE
minOfsMAPE = 20
for i in range(1,rangeMax):
    if listOfsMAPE[i] < minOfsMAPE:
        minOfsMAPE = listOfsMAPE[i]
k = listOfsMAPE.index(minOfsMAPE)
print("the best k is", k)

regressor = KNeighborsRegressor(n_neighbors = k)
forecaster = ReducedForecaster(
    regressor, scitype = "regressor", window_length = 15, strategy = "recursive"
)
forecaster.fit(y_train)
y_pred_kNN_bestk = forecaster.predict(fh)
print(y_test)
print(y_pred_kNN_bestk)
# loss4 = smape_loss(y_test, y_pred_kNN_bestk)
# print("The best sMAPE loss for kNN method is obtained when k =", 1, ", which is:", loss4)
fig4, ax4 = plot_series(y_train, y_test, y_pred_kNN_bestk, labels = ["y_train", "y_test", "y_pred"])
ax4.set_title("Prediction with kNR optimized")
plt.show()
# plot and zoom in the test set
fig4bis, ax4bis = plot_series(y_test, y_pred_kNN_bestk.drop(y_pred_kNN_bestk.index[0]), labels = ["y_test", "y_pred"])
ax4bis.set_title("The Same result zoomed in to the test set y_test")
plt.show()

# plot the curve of sMAPE - k
listOfsMAPE[0] = listOfsMAPE[1]
plt.figure(2)
plt.plot(range(0, rangeMax), listOfsMAPE)
plt.title("sMPAE-k with k is the length of the forecasting window")
plt.show()



'''
# predicting with ExponentialSmoothing
listOfsMAPE_ES = []
for spTrial in range(1,54):
    forecaster = ExponentialSmoothing(trend = None, seasonal = None, sp = spTrial)
    forecaster.fit(y_train)
    y_pred_withES = forecaster.predict(fh)

    y_pred_withES = y_pred_withES.drop(y_pred_withES.index[0])
    loss5 = smape_loss(y_test, y_pred_withES)
    listOfsMAPE_ES.append(loss5)
# search the min of sMAPE
minOfsMAPE = 20
for i in range(1, len(listOfsMAPE_ES)):
    if listOfsMAPE_ES[i] < minOfsMAPE:
        minOfsMAPE = listOfsMAPE_ES[i]
sptOptimal = listOfsMAPE_ES.index(minOfsMAPE)
print("The best sp for Exponential Smoothing method is:", sptOptimal+1)
print("The corresponding sMAPE is :", listOfsMAPE_ES[sptOptimal])

forecaster = ExponentialSmoothing(trend = None, seasonal = None, sp = sptOptimal+1)
forecaster.fit(y_train)
y_pred_withES = forecaster.predict(fh)
fig5, ax5 = plot_series(y_test, y_pred_withES, labels = ["y_test", "y_pred"])
ax5.set_title("Exponantial Smooting")
plt.show()
'''

'''
# prediction with autoArima
# didn't get the result, it takes too much time to train the model
forecaster = AutoARIMA(sp = 60, suppress_warnings = True)
forecaster.fit(y_train)
y_pred_withAutoArima = forecaster.predict(fh)
fig6, ax6 = plot_series(y_train, y_test, y_pred_withAutoArima, labels = ["y_train", "y_test", "y_pred"])
ax6.set_title("autoArima")
loss6 = smape_loss(y_test, y_pred_withAutoArima)
print("The sMAPE for auto-Arima method is:", loss6)
'''

'''
# prediction with single Arima
forecaster = ARIMA(
    order = (1, 1, 2), seasonal_order = (1, 1, 1, 54), suppress_warnings = True
)
forecaster.fit(y_train)
y_pred_singleArima = forecaster.predict(fh)
# print("Method single Arima : y_train:", y_train)
# print("Method single Arima : y_test:", y_test)
# print("Method single Arima : y_pred:", y_pred_withES)
# the result is ridiculously bad, it presents a trend of decrease
fig7, ax7 = plot_series(y_test, y_pred_singleArima, labels = ["y_test", "y_pred"])
ax7.set_title("Arima")
plt.show()
y_pred_singleArima = y_pred_singleArima.drop(y_pred_singleArima.index[0])
loss7 = smape_loss(y_test, y_pred_singleArima)
print("The sMAPE for single-Arima method is:", loss7)
'''

'''
# prediction with BATS
# This method runs relatively slow and it produces an outcome similar to mean value prediction
forecaster = BATS(sp=7, use_trend=True, use_box_cox=False)
forecaster.fit(y_train)
y_pred_BATS = forecaster.predict(fh)
fig8, ax8 = plot_series(y_test, y_pred_BATS, labels=["y_test", "y_pred"])
plt.show()
y_pred_BATS = y_pred_BATS.drop(y_pred_BATS.index[0])
loss8 = smape_loss(y_test, y_pred_BATS)
print("The sMAPE for BATS method is:", loss8)
'''
'''
# prediction with TBATS
forecaster = TBATS(sp=12, use_trend=True, use_box_cox=False)
forecaster.fit(y_train)
y_pred_TBATS = forecaster.predict(fh)
fig9, ax9 = plot_series(y_test, y_pred_TBATS, labels=["y_test", "y_pred"])
ax9.set_title(TBATS)
plt.show()
y_pred_TBATS = y_pred_TBATS.drop(y_pred_TBATS.index[0])
loss9 = smape_loss(y_test, y_pred_TBATS)
print("The sMAPE for TBATS method is:", loss9)
'''

'''
# prediction with autoETS
# modify the data, replacing 0 by 0.01
# change all dato into float
y = pd.Series(data = df['entries_daily_0_modified'])
y_train, y_test = temporal_train_test_split(y, test_size = 42)
forecaster = AutoETS(error = None, trend = None, sp = 52, auto = True)
forecaster.fit(y_train)
y_pred_autoETS = forecaster.predict(fh)
fig10, ax10 = plot_series(y_test, y_pred_autoETS, labels = ["y_test", "y_pred"])
plt.show()
y_pred_autoETS = y_pred_autoETS.drop(y_pred_autoETS.index[0])
loss10 = smape_loss(y_test, y_pred_autoETS)
print("The sMAPE for autoETS method is:", loss10)
'''



# Helper functions, some other possible metrics for evaluations
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
