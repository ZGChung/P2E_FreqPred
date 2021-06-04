from warnings import simplefilter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sktime.datasets import load_airline
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import (
    EnsembleForecaster,
    # MultiplexForecaster,
    ReducedForecaster,
    TransformedTargetForecaster,

)
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import (
    ExpandingWindowSplitter,
    ForecastingGridSearchCV,
    SlidingWindowSplitter,
    temporal_train_test_split,
)
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.performance_metrics.forecasting import sMAPE, smape_loss
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.utils.plotting import plot_series

simplefilter("ignore", FutureWarning)

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
print(df)
y = pd.Series(data = df['entries_daily'])
print(y)
y2 = load_airline()
print(y2)
# Use the data of 2019 as training set, marked in blue in the plot
# Use the data pre-COVID of 2020 as testing set, marked in orange in the plot
fig1, ax1 = plot_series(y);
plt.close()
y_train, y_test = temporal_train_test_split(y, test_size = 74)
fig2, ax2 = plot_series(y_train, y_test, labels = ["y=train", "y=test"])
ax2.set_title("Original data after Train-Test separation")
# plt.close()
# print(y_train.shape[0], y_test.shape[0])
# use a forecasting horizon the same size as the test set
fh = np.arange(len(y_test)) + 1
print(fh)
plt.show()
