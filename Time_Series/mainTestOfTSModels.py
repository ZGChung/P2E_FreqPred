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
df = pd.read_csv("data_daily_preCOVID_2cols.csv")

# # test read file
# print(df.shape)
# print(df.info())
# df.head(5)

# convert the first column to datetime format
# df['time'] = pd.to_datetime(df['time'], unit = 's')
# print(df)
# df = df.set_index('time')
y = pd.Series(data = df['entries_daily'])
# x = df.time
# y = df.entries_daily

# Use the data of 2019 as training set, marked in blue in the plot
# Use the data pre-COVID of 2020 as testing set, marked in orange in the plot
fig1, ax1 = plot_series(y);
plt.close()
y_train, y_test = temporal_train_test_split(y, test_size = 74)
fig2, ax2 = plot_series(y_train, y_test, labels = ["y=train", "y=test"])
ax2.set_title("Original data after Train-Test separation")
plt.close()
# print(y_train.shape[0], y_test.shape[0])
# use a forecasting horizon the same size as the test set
fh = np.arange(len(y_test)) + 1
# print(fh)

# predicting with the last value
# a naive test just to verify the model works
forecaster = NaiveForecaster(strategy = "last")
forecaster.fit(y_train)
y_pred_NaiveForecaster = forecaster.predict(fh)
fig3, ax3 = plot_series(y_train, y_test, y_pred_NaiveForecaster, labels = ["y_train", "y_test", "y_pred"])
plt.close()
# we use sMAPE as the evaluation metric here
# sMAPE represents: symmetric Mean Absolute Percentage Error
loss3 = smape_loss(y_pred_NaiveForecaster, y_test)
print("The sMAPE for NaiveForecaster method is:", loss3)

# predicting with kNN
# # search the k for the kNN minimizing the sMAPE
# listOfsMAPE = []
# listOfsMAPE.append(20)  # initialize the first as a big number
# for i in range(1,351):
#     regressor = KNeighborsRegressor(n_neighbors = i)
#     forecaster = ReducedForecaster(
#         regressor, scitype = "regressor", window_length = 15, strategy = "recursive"
#     )
#     forecaster.fit(y_train)
#     y_pred = forecaster.predict(fh)
#     loss = smape_loss(y_test, y_pred)
#     print("The sMAPE loss for ", i,"NN prediction is:", loss)
#     listOfsMAPE.append(loss)
# # search the min of sMAPE
# minOfsMAPE = 20
# for i in range(1,351):
#     if listOfsMAPE[i] < minOfsMAPE:
#         minOfsMAPE = listOfsMAPE[i]
# k = listOfsMAPE.index(minOfsMAPE)

# note: the best k is 350, which is in fact the sup of our range of trial
# note2: k = 350 predicts in fact the mean value
# ntoe3: k = 350 and k = 1 gives quite similar sMAPE but the latter is more reasonable
regressor = KNeighborsRegressor(n_neighbors = 1)
forecaster = ReducedForecaster(
    regressor, scitype = "regressor", window_length = 15, strategy = "recursive"
)
forecaster.fit(y_train)
y_pred_kNN_bestk = forecaster.predict(fh)
loss4 = smape_loss(y_test, y_pred_kNN_bestk)
print("The best sMAPE loss for kNN method is obtained when k =", 1, ", which is:", loss4)
fig4, ax4 = plot_series(y_train, y_test, y_pred_kNN_bestk, labels = ["y_train", "y_test", "y_pred"])
ax4.set_title("Prediction with kNN optimized")
plt.close()
# plot and zoom in the test set
fig4bis, ax4bis = plot_series(y_test, y_pred_kNN_bestk, labels = ["y_test", "y_pred"])
plt.close()

# # plot the curve of sMAPE - k
# listOfsMAPE[0] = listOfsMAPE[1]
# plt.figure(2)
# plt.plot(range(0, 351), listOfsMAPE)
# plt.title("sMPAE-k with k for the kNN")

# predicting with ExponentialSmoothing
forecaster = ExponentialSmoothing(trend = "add", seasonal = "add", sp = 60)
forecaster.fit(y_train)
y_pred_withES = forecaster.predict(fh)
# print("Method ES : y_train:", y_train)
# print("Method ES : y_test:", y_test)
# print("Method ES : y_pred:", y_pred_withES)
# I didn't plot the result of ES because we even got some negative predictions
# as a result the function plot_series cannot function normally
# one possible explanation is that we cannot really identify a periodicity smaller than a year here
fig5, ax5 = plot_series(y_test, y_pred_withES, labels = ["y_test", "y_pred"])
plt.close()
loss5 = smape_loss(y_test, y_pred_withES)
print("The sMAPE for Exponential Smoothing method is:", loss5)

# prediction with autoArima
forecaster = AutoARIMA(sp = 60, suppress_warnings = True)
forecaster.fit(y_train)
y_pred_withAutoArima = forecaster.predict(fh)
fig6, ax6 = plot_series(y_train, y_test, y_pred_withAutoArima, labels = ["y_train", "y_test", "y_pred"])
loss6 = smape_loss(y_test, y_pred_withAutoArima)
print("The sMAPE for auto-Arima method is:", loss6)

# prediction with single Arima
forecaster = ARIMA(
    order = (1, 1, 0), seasonal_order = (0, 1, 0, 12), suppress_warnings = True
)
forecaster.fit(y_train)
y_pred_singleArima = forecaster.predict(fh)
# print("Method single Arima : y_train:", y_train)
# print("Method single Arima : y_test:", y_test)
# print("Method single Arima : y_pred:", y_pred_withES)
# the result is ridiculously bad, it presents a trend of decrease
fig7, ax7 = plot_series(y_test, y_pred_singleArima, labels = ["y_test", "y_pred"])
plt.close()
loss7 = smape_loss(y_test, y_pred_singleArima)
print("The sMAPE for single-Arima method is:", loss7)

# # prediction with BATS
# # This method runs relatively slow and it produces an outcome similar to mean value prediction
# forecaster = BATS(sp=60, use_trend=True, use_box_cox=False)
# forecaster.fit(y_train)
# y_pred_BATS = forecaster.predict(fh)
# fig8, ax8 = plot_series(y_test, y_pred_BATS, labels=["y_test", "y_pred"])
# loss8 = smape_loss(y_test, y_pred_BATS)
# print("The sMAPE for BATS method is:", loss8)

# # prediction with TBATS
# forecaster = TBATS(sp=12, use_trend=True, use_box_cox=False)
# forecaster.fit(y_train)
# y_pred_TBATS = forecaster.predict(fh)
# fig9, ax9 = plot_series(y_test, y_pred_TBATS, labels=["y_test", "y_pred"])
# plt.close()
# loss9 = smape_loss(y_test, y_pred_TBATS)
# print("The sMAPE for TBATS method is:", loss9)

# prediction with autoETS
# modify the data, replacing 0 by 0.01
# change all dato into float
y = pd.Series(data = df['entries_daily_0_modified'])
y_train, y_test = temporal_train_test_split(y, test_size = 74)
forecaster = AutoETS(error = "add", trend = None, sp = 7, auto = True)
forecaster.fit(y_train)
y_pred_autoETS = forecaster.predict(fh)
fig10, ax10 = plot_series(y_test, y_pred_autoETS, labels = ["y_test", "y_pred"])
plt.close()
loss10 = smape_loss(y_test, y_pred_autoETS)
print("The sMAPE for autoETS method is:", loss10)

# save the figures and plot them in the same plot
fig1.savefig('Prediction_Results/fig1.png')
fig2.savefig('Prediction_Results/fig2.png')
fig3.savefig('Prediction_Results/fig3.png')
fig4.savefig('Prediction_Results/fig4.png')

# for i in range(NumberOfPredictions):
#     plt.close()

img1 = mpimg.imread('Prediction_Results/fig1.png')
img2 = mpimg.imread('Prediction_Results/fig2.png')
img3 = mpimg.imread('Prediction_Results/fig3.png')
img4 = mpimg.imread('Prediction_Results/fig4.png')

# fig = plt.figure()
# ax = fig.add_subplot(4, 1, 1)
# ax.set_title('Original Data')
# plt.imshow(img1)
# ax = fig.add_subplot(4, 1, 1)
# ax.set_title('Original data after separation')
# plt.imshow(img2)
# ax = fig.add_subplot(4, 1, 2)
# ax.set_title('Prediction with last value')
# plt.imshow(img3)
plt.show()

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
