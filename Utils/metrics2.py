import numpy as np
from numpy.random import seed
seed(1)

import pandas as pd
from math import sqrt
from scipy.integrate import quad


tolerance = 1E-8

########################
# UTILITY MODELS
########################

def detrend(insample_data):
  """
  Calculates a & b parameters of LRL
  :param insample_data:
  :return:
  """
  x = np.arange(len(insample_data))
  a, b = np.polyfit(x, insample_data, 1)
  return a, b

def deseasonalize(original_ts, ppy):
  """
  Calculates and returns seasonal indices
  :param original_ts: original data
  :param ppy: periods per year
  :return:
  """
  """
  # === get in-sample data
  original_ts = original_ts[:-out_of_sample]
  """
  if seasonality_test(original_ts, ppy):
    # ==== get moving averages
    ma_ts = moving_averages(original_ts, ppy)

    # ==== get seasonality indices
    le_ts = original_ts * 100 / ma_ts
    le_ts = np.hstack((le_ts, np.full((ppy - (len(le_ts) % ppy)), np.nan)))
    le_ts = np.reshape(le_ts, (-1, ppy))
    si = np.nanmean(le_ts, 0)
    norm = np.sum(si) / (ppy * 100)
    si = si / norm
  else:
    si = np.ones(ppy)

  return si

def moving_averages(ts_init, window):
  """
  Calculates the moving averages for a given TS
  :param ts_init: the original time series
  :param window: window length
  :return: moving averages ts
  """
  """
  As noted by Professor Isidro Lloret Galiana:
  line 82:
  if len(ts_init) % 2 == 0:
  
  should be changed to
  if window % 2 == 0:
  
  This change has a minor (less then 0.05%) impact on the calculations of the seasonal indices
  In order for the results to be fully replicable this change is not incorporated into the code below
  """
  ts_init = pd.Series(ts_init)
  
  if len(ts_init) % 2 == 0:
    ts_ma = ts_init.rolling(window, center=True).mean()
    ts_ma = ts_ma.rolling(2, center=True).mean()
    ts_ma = np.roll(ts_ma, -1)
  else:
    ts_ma = ts_init.rolling(window, center=True).mean()

  return ts_ma

def seasonality_test(original_ts, ppy):
  """
  Seasonality test
  :param original_ts: time series
  :param ppy: periods per year
  :return: boolean value: whether the TS is seasonal
  """
  s = acf(original_ts, 1)
  for i in range(2, ppy):
    s = s + (acf(original_ts, i) ** 2)

  limit = 1.645 * (sqrt((1 + 2 * s) / len(original_ts)))

  return (abs(acf(original_ts, ppy))) > limit

def acf(data, k):
  """
  Autocorrelation function
  :param data: time series
  :param k: lag
  :return:
  """
  m = np.mean(data)
  s1 = 0
  for i in range(k, len(data)):
    s1 = s1 + ((data[i] - m) * (data[i - k] - m))

  s2 = 0
  for i in range(0, len(data)):
    s2 = s2 + ((data[i] - m) ** 2)

  return float(s1 / s2)

class Naive:
  """
  Naive model.
  This benchmark model produces a forecast that is equal to
  the last observed value for a given time series.
  """
  def __init__(self):
    pass
  
  def fit(self, ts_init):
    """
    ts_init: the original time series
    ts_naive: last observations of time series
    """
    self.ts_naive = [ts_init[-1]]
    return self

  def predict(self, h):
    return np.array(self.ts_naive * h)
    
class SeasonalNaive:
  """
  Seasonal Naive model.
  This benchmark model produces a forecast that is equal to
  the last observed value of the same season for a given time 
  series.
  """
  def __init__(self):
    pass
  
  def fit(self, ts_init, seasonality):
    """
    ts_init: the original time series
    frcy: frequency of the time series
    ts_naive: last observations of time series
    """
    self.ts_seasonal_naive = ts_init[-seasonality:]
    return self

  def predict(self, h):
    repetitions = int(np.ceil(h/len(self.ts_seasonal_naive)))
    y_hat = np.tile(self.ts_seasonal_naive, reps=repetitions)[:h]
    return y_hat

class Naive2:
  """
  Naive2 model.
  Popular benchmark model for time series forecasting that automatically adapts
  to the potential seasonality of a series based on an autocorrelation test.
  If the series is seasonal the model composes the predictions of Naive and SeasonalNaive,
  else the model predicts on the simple Naive.
  """
  def __init__(self, seasonality):
    self.seasonality = seasonality
    
  def fit(self, ts_init):
    seasonality_in = deseasonalize(ts_init, ppy=self.seasonality)
    windows = int(np.ceil(len(ts_init) / self.seasonality))
    
    self.ts_init = ts_init
    self.s_hat = np.tile(seasonality_in, reps=windows)[:len(ts_init)]
    self.ts_des = ts_init / self.s_hat
            
    return self
    
  def predict(self, h):
    s_hat = SeasonalNaive().fit(self.s_hat,
                                seasonality=self.seasonality).predict(h)
    r_hat = Naive().fit(self.ts_des).predict(h)        
    y_hat = s_hat * r_hat
    return y_hat

########################
# METRICS
########################



def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def _percentage_error(actual: np.ndarray, predicted: np.ndarray):
    """ Percentage error """
    return (actual - predicted) / (actual+1E-7)

def rmdspe(actual: np.ndarray, predicted: np.ndarray):
    """
    Root Median Squared Percentage Error
    Note: result is NOT multiplied by 100
    """
    return np.sqrt(np.median(np.square(_percentage_error(actual, predicted))))


def NAPE(actual: np.ndarray, predicted: np.ndarray):
    """ Normalized Absolute Percentage Error """
    __mape = mape(actual, predicted)
    per_error = _percentage_error(actual, predicted)
    
    all_sumed = np.sum(np.square(per_error - __mape))

    return np.sqrt(all_sumed)/(len(actual) - 1)


### new metric

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / (true + tolerance)))


def MASE(pred, true, naive_pred):
   return MAE(pred, true)/ MAE(naive_pred, true)

def QuantileLoss(target, forecast, q=0.5):
    return (2*np.mean(np.abs((forecast-target)*((target<=forecast)-q))))



def quantile_loss(y, y_pred, q=0.5):
  residual = y_pred - y
  loss = np.max((q - 1) * residual, q * residual)
  return np.mean(loss)

## coverage
def coverage(target, forecast):
  return (np.mean((target < forecast)))


def msis(target,lower_quantile,upper_quantile,seasonal_error,alpha=0.75):
  """alpha - significance level"""
  numerator = np.mean(
      upper_quantile-lower_quantile
      + 2.0/ alpha* (lower_quantile - target)*(target < lower_quantile)
      + 2.0/ alpha*(target - upper_quantile)*(target > upper_quantile)
  )
  return (numerator/seasonal_error)


def mape(y, y_hat):
  """
  Calculates Mean Absolute Percentage Error.

  Parameters
  ----------
  y: numpy array
    actual test values
  y_hat: numpy array
    predicted values
  
  Returns
  -------
  mape: float
    mean absolute percentage error
  """
  y = np.reshape(y, (-1,))
  y_hat = np.reshape(y_hat, (-1,))
  mape = np.mean(np.abs(y - y_hat) / np.abs(y))
  return mape

def smape(y, y_hat):
  """
  Calculates Symmetric Mean Absolute Percentage Error.

  Parameters
  ----------  
  y: numpy array
    actual test values
  y_hat: numpy array
    predicted values
  
  Returns
  -------
  smape: float
    symmetric mean absolute percentage error
  """
  y = np.reshape(y, (-1,))
  y_hat = np.reshape(y_hat, (-1,))
  smape = np.mean(2.0 * np.abs(y - y_hat) / (np.abs(y) + np.abs(y_hat) + tolerance))
  return smape

def mase(y, y_hat, y_train, seasonality=24):
  """
  Calculates Mean Absolute Scaled Error.

  Parameters
  ----------
  y: numpy array
    actual test values
  y_hat: numpy array
    predicted values
  y_train: numpy array
    actual train values for Naive1 predictions
  seasonality: int
    main frequency of the time series
    Quarterly 4, Daily 7, Monthly 12
  
  Returns
  -------
  mase: float
    mean absolute scaled error
  """
  y_hat_naive = []
  for i in range(seasonality, len(y_train)):
      y_hat_naive.append(y_train[(i - seasonality)])

  masep = np.mean(abs(y_train[seasonality:] - y_hat_naive))
  mase = np.mean(abs(y - y_hat)) / masep
  return mase


def crps_score(y_true, y_pred):
    def integrand(x):
        return (y_pred - (x >= y_true)) ** 2

    crps, _ = quad(integrand, -np.inf, np.inf)
    return crps




########################
# PANEL EVALUATION
########################

def evaluate_panel(y_panel, y_hat_panel, metric,
                   y_insample=None, seasonality=None):
  """
  Calculates metric for y_panel and y_hat_panel
  y_panel: pandas df
    panel with columns unique_id, ds, y
  y_naive2_panel: pandas df
    panel with columns unique_id, ds, y_hat
  y_insample: pandas df
    panel with columns unique_id, ds, y (train)
    this is used in the MASE
  seasonality: int
    main frequency of the time series
    Quarterly 4, Daily 7, Monthly 12
  return: list of metric evaluations
  """
  metric_name = metric.__code__.co_name

  y_panel = y_panel.sort_values(['unique_id', 'ds'])
  y_hat_panel = y_hat_panel.sort_values(['unique_id', 'ds'])
  if y_insample is not None:
      y_insample = y_insample.sort_values(['unique_id', 'ds'])

  assert len(y_panel)==len(y_hat_panel)
  assert all(y_panel.unique_id.unique() == y_hat_panel.unique_id.unique()), "not same u_ids"

  evaluation = []
  for u_id in y_panel.unique_id.unique():
    top_row = np.asscalar(y_panel['unique_id'].searchsorted(u_id, 'left'))
    bottom_row = np.asscalar(y_panel['unique_id'].searchsorted(u_id, 'right'))
    y_id = y_panel[top_row:bottom_row].y.to_numpy()

    top_row = np.asscalar(y_hat_panel['unique_id'].searchsorted(u_id, 'left'))
    bottom_row = np.asscalar(y_hat_panel['unique_id'].searchsorted(u_id, 'right'))
    y_hat_id = y_hat_panel[top_row:bottom_row].y_hat.to_numpy()
    assert len(y_id)==len(y_hat_id)

    if metric_name == 'mase':
      assert (y_insample is not None) and (seasonality is not None)
      top_row = np.asscalar(y_insample['unique_id'].searchsorted(u_id, 'left'))
      bottom_row = np.asscalar(y_insample['unique_id'].searchsorted(u_id, 'right'))
      y_insample_id = y_insample[top_row:bottom_row].y.to_numpy()
      evaluation_id = metric(y_id, y_hat_id, y_insample_id, seasonality)
    else:
      evaluation_id = metric(y_id, y_hat_id)
    evaluation.append(evaluation_id)
  return evaluation

def owa(y_panel, y_hat_panel, y_naive2_panel, y_insample, seasonality):
  """
  Calculates MASE, sMAPE for Naive2 and current model
  then calculatess Overall Weighted Average.
  y_panel: pandas df
    panel with columns unique_id, ds, y
  y_hat_panel: pandas df
    panel with columns unique_id, ds, y_hat
  y_naive2_panel: pandas df
    panel with columns unique_id, ds, y_hat
  y_insample: pandas df
    panel with columns unique_id, ds, y (train)
    this is used in the MASE
  seasonality: int
    main frequency of the time series
    Quarterly 4, Daily 7, Monthly 12
  return: OWA
  """
  total_mase = evaluate_panel(y_panel, y_hat_panel, mase, 
                              y_insample, seasonality)
  total_mase_naive2 = evaluate_panel(y_panel, y_naive2_panel, mase,
                                     y_insample, seasonality)
  total_smape = evaluate_panel(y_panel, y_hat_panel, smape)
  total_smape_naive2 = evaluate_panel(y_panel, y_naive2_panel, smape)

  assert len(total_mase) == len(total_mase_naive2)
  assert len(total_smape) == len(total_smape_naive2)
  assert len(total_mase) == len(total_smape)
  
  naive2_mase = np.mean(total_mase_naive2)
  naive2_smape = np.mean(total_smape_naive2) * 100

  model_mase = np.mean(total_mase)
  model_smape = np.mean(total_smape) * 100
  
  model_owa = ((model_mase/naive2_mase) + (model_smape/naive2_smape))/2
  return model_owa, model_mase, model_smape

def evaluate_prediction_owa(y_hat_df, y_train_df, y_test_df, 
                            naive2_seasonality):
    """
    y_hat_df: pandas df
      panel with columns unique_id, ds, y_hat
    y_train_df: pandas df
      panel with columns unique_id, ds, y
    y_test_df: pandas df
      panel with columns unique_id, ds, y, y_hat_naive2
    naive2_seasonality: int
      seasonality for the Naive2 predictions (needed for owa)
    model: python class
      python class with predict method
    """
    y_panel = y_test_df.filter(['unique_id', 'ds', 'y'])
    y_naive2_panel = y_test_df.filter(['unique_id', 'ds', 'y_hat_naive2'])
    y_naive2_panel.rename(columns={'y_hat_naive2': 'y_hat'}, inplace=True)
    y_hat_panel = y_hat_df
    y_insample = y_train_df.filter(['unique_id', 'ds', 'y'])

    model_owa, model_mase, model_smape = owa(y_panel, y_hat_panel, 
                                             y_naive2_panel, y_insample,
                                             seasonality=naive2_seasonality)

    print(15*'=', ' Model evaluation ', 14*'=')
    print('OWA: {} '.format(np.round(model_owa, 3)))
    print('SMAPE: {} '.format(np.round(model_smape, 3)))
    print('MASE: {} '.format(np.round(model_mase, 3)))
    return model_owa, model_mase, model_smape



def get_metric(pred, true, naive_pred, pred_len=24):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    # nape = NAPE(pred, true)
    _rmdspe =  rmdspe(true, pred)

    ### new metric

    mape = MAPE(pred, true)
    _smape = smape(true, pred)
    _mase = MASE(true, pred, naive_pred)

    Q25 = QuantileLoss(true, pred, q=0.25)
    Q75 = QuantileLoss(true, pred, q=0.75)
    
    # for k in [pred_len//8, pred_len//4, pred_len//2]:
    #   _acf = acf(true -pred, k)
    #   print('k = %d, acf = {:.4f}',k, _acf)
    
    crps = []
    for i in range(1,10):
      crps.append(QuantileLoss(true, pred, q=i/10))
    crps = np.mean(crps)

    return mae, mse, rmse, _rmdspe, mape, _smape, _mase, Q25, Q75, crps