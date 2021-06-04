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

print("Hello world! Program begins.")
df1 = pd.read_csv("data_daily_dates.csv")
# df2 = pd.read_csv("data_daily origin.csv")
# y = pd.Series(data = df2['entries_daily'])
# fig1, ax1 = plot_series(y)
# plt.show()
# test read file
print("df1.shape", df1.shape)
print("df1", df1)

# df = df1.loc[df1["entries_daily"] != 0]
# df = df.reset_index()
# print("df.shape", df.shape)
# print("df", df)


# print("HHHHH", df1.iloc[3,1])

# convert the first column to datetime format
# df1['time'] = pd.to_datetime(df1['time'], unit = 's')
# df1 = df1.set_index('time')
# print(df1)
# df1.to_csv("data_daily_dates.csv")
y = pd.Series(data = df1['entries_daily'])
# x = df.time
# y = df.entries_daily


# Use the data of 2019 as training set, marked in blue in the plot
# Use the data pre-COVID of 2020 as testing set, marked in orange in the plot
fig1, ax1 = plot_series(y)
plt.show()
y_train, y_test = temporal_train_test_split(y, test_size = 28)
fig2, ax2 = plot_series(y_train, y_test, labels = ["y=train", "y=test"])
ax2.set_title("Original data after Train-Test separation")
plt.show()
# print(y_train.shape[0], y_test.shape[0])
# use a forecasting horizon the same size as the test set
fh = np.arange(len(y_test)+1)
# print(fh)

# predicting with kNN

# search the k for the kNN minimizing the sMAPE
listOfsMAPE = []
'''
listOfsMAPE.append(20)  # initialize the first as a big number
for i in range(1,351):
    regressor = KNeighborsRegressor(n_neighbors = i)
    forecaster = ReducedForecaster(
        regressor, scitype = "regressor", window_length = 15, strategy = "recursive"
    )
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    loss = smape_loss(y_test, y_pred)
    print("The sMAPE loss for ", i,"NN prediction is:", loss)
    listOfsMAPE.append(loss)
# search the min of sMAPE
minOfsMAPE = 20
for i in range(1,351):
    if listOfsMAPE[i] < minOfsMAPE:
        minOfsMAPE = listOfsMAPE[i]
k = listOfsMAPE.index(minOfsMAPE)

# note: the best k is 350, which is in fact the sup of our range of trial
# note2: k = 350 predicts in fact the mean value
# ntoe3: k = 350 and k = 1 gives quite similar sMAPE but the latter is more reasonable
'''

regressor = KNeighborsRegressor(n_neighbors = 1)
forecaster = ReducedForecaster(
    regressor, scitype = "regressor", window_length = 15, strategy = "recursive"
)
forecaster.fit(y_train)
y_pred_kNN_bestk = forecaster.predict(fh)
# loss4 = smape_loss(y_test, y_pred_kNN_bestk)
# print("The best sMAPE loss for kNN method is obtained when k =", 1, ", which is:", loss4)
fig4, ax4 = plot_series(y_train, y_test, y_pred_kNN_bestk, labels = ["y_train", "y_test", "y_pred"])
ax4.set_title("Prediction with kNN optimized")
plt.show()
# plot and zoom in the test set
fig4bis, ax4bis = plot_series(y_test, y_pred_kNN_bestk, labels = ["y_test", "y_pred"])
plt.show()

# plot the curve of sMAPE - k
listOfsMAPE[0] = listOfsMAPE[1]
plt.figure(2)
plt.plot(range(0, 351), listOfsMAPE)
plt.show()
plt.title("sMPAE-k with k for the kNR")
