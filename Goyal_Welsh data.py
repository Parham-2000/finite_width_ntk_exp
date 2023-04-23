import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import os


# AND NOW WE START WORKING WITH REAL DATA

goyal_welch_data = pd.read_csv(os.path.join(folder, 'PredictorData2021.csv'), index_col=0)
goyal_welch_data.index = pd.to_datetime(goyal_welch_data.index, format='%Y%m')

for column in goyal_welch_data.columns:
    goyal_welch_data[column] = [float(str(x).replace(',', '')) for x in goyal_welch_data[column]]


import sys

sys.path.append("/content/gdrive/My Drive/colab_env/lib/python3.8/site-packages")
import statsmodels.api as sm



# Now we read the data we will be using for our experiments. As before, we are investigating he classic dataset from [A Comprehensive Look at the Empirical Performance of Equity Premium Prediction](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=517667) and implemented in [The Virtue of Complexity in Return Prediction](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3984925)

goyal_welch_data.Rfree.iloc[-12:] = goyal_welch_data.Rfree.iloc[-12:] / 12
goyal_welch_data['returns'] = (
            (goyal_welch_data.Index + 0 * goyal_welch_data.D12 / 12) / goyal_welch_data.Index.shift(1) - 1).fillna(0)
goyal_welch_data['excess_returns'] = goyal_welch_data.returns - goyal_welch_data.Rfree
cleaned_data = goyal_welch_data.loc['1975':].drop(columns=['csp']).fillna(0)
signal_columns = ['Index', 'D12', 'E12', 'b/m', 'tbl', 'AAA', 'BAA', 'lty', 'ntis',
                  'Rfree', 'infl', 'ltr', 'corpr', 'svar']
# the next is very important: we are lagging data by 1 month so that we are actually predicting future
data_for_signals = cleaned_data[signal_columns].shift(1).fillna(0)
labels = cleaned_data.excess_returns.values.reshape(-1, 1)
signals = data_for_signals.values
data_for_signals.shape, data_for_signals.columns


def normalize(data: np.ndarray,
              ready_normalization: dict = None):
    if ready_normalization is None:
        data_std = data.std(0)
        # data = data / data_std

        data_max = np.max(data, axis=0)
        data_min = np.min(data, axis=0)
    else:
        data_std = ready_normalization['std']
        # data = data / data_std

        data_max = ready_normalization['max']
        data_min = ready_normalization['min']

    data = data - data_min
    data = data / (data_max - data_min)
    data = data - 0.5
    normalization = {'std': data_std,
                     'max': data_max,
                     'min': data_min}
    return data, normalization


normalize_raw_data = True
cheat_and_use_future_data = False

split = int(signals.shape[0] / 2)
train_labels = labels[:split]
test_labels = labels[split:]
dates = data_for_signals.index[split:]

if normalize_raw_data:
    signals[:split, :], normalization = normalize(signals[:split])
    if cheat_and_use_future_data:
        signals[split:, :] = normalize(signals[split:, :])[0]
    else:
        signals[split:, :] = normalize(signals[split:, :],
                                       ready_normalization=normalization)[0]


def sharpe_ratio(x):
    return np.round(np.sqrt(12) * x.mean(0) / x.std(0), 2)


def ridge_regr(signals: np.ndarray,
               labels: np.ndarray,
               future_signals: np.ndarray,
               shrinkage_list: np.ndarray):
    """
    Regression is
    beta = (zI + S'S/t)^{-1}S'y/t = S' (zI+SS'/t)^{-1}y/t
    Inverting matrices is costly, so we use eigenvalue decomposition:
    (zI+A)^{-1} = U (zI+D)^{-1} U' where UDU' = A is eigenvalue decomposition,
    and we use the fact that D @ B = (diag(D) * B) for diagonal D, which saves a lot of compute cost
    :param signals: S
    :param labels: y
    :param future_signals: out of sample y
    :param shrinkage_list: list of ridge parameters
    :return:
    """
    t_ = signals.shape[0]
    p_ = signals.shape[1]
    if p_ < t_:
        # this is standard regression
        eigenvalues, eigenvectors = np.linalg.eigh(signals.T @ signals / t_)
        means = signals.T @ labels.reshape(-1, 1) / t_
        multiplied = eigenvectors.T @ means
        intermed = np.concatenate([(1 / (eigenvalues.reshape(-1, 1) + z)) * multiplied for z in shrinkage_list],
                                  axis=1)
        betas = eigenvectors @ intermed
    else:
        # this is the weird over-parametrized regime
        eigenvalues, eigenvectors = np.linalg.eigh(signals @ signals.T / t_)
        means = labels.reshape(-1, 1) / t_
        multiplied = eigenvectors.T @ means
        intermed = np.concatenate([(1 / (eigenvalues.reshape(-1, 1) + z)) * multiplied for z in shrinkage_list],
                                  axis=1)
        tmp = eigenvectors.T @ signals
        betas = tmp.T @ intermed
    predictions = future_signals @ betas
    return betas, predictions


def regression_with_tstats(predicted_variable, explanatory_variables):
    '''
    This function gets t-stats from regression
    :param predicted_variable:
    :param explanatory_variables:
    :return:
    '''
    x_ = explanatory_variables
    x_ = sm.add_constant(x_)
    y_ = predicted_variable
    # Newey-West standard errors with maxlags
    z_ = x_.copy().astype(float)
    result = sm.OLS(y_, z_).fit(cov_type='HAC', cov_kwds={'maxlags': 10})
    try:
        tstat = np.round(result.summary2().tables[1]['z'], 1)  # alpha t-stat (because for 'const')
        tstat.index = list(z_.columns)
    except:
        print(f'something is wrong for t-stats')
    return tstat


shrinkage_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

beta_estimate_using_train_sample, linear_oos_preds = ridge_regr(signals=signals[:split, :],
                                                                labels=train_labels,
                                                                future_signals=signals[split:, :],
                                                                shrinkage_list=shrinkage_list)
linear_oos_preds = pd.DataFrame(linear_oos_preds, index=cleaned_data.index[split:], columns=shrinkage_list)
linear_kitchen_sink_market_timing_returns = linear_oos_preds * test_labels.reshape(-1, 1)
cleaned_data = pd.concat([cleaned_data, linear_kitchen_sink_market_timing_returns], axis=1)
tmp = cleaned_data[['excess_returns'] + shrinkage_list].iloc[split:]
tmp = tmp / tmp.std()
sr = sharpe_ratio(tmp)
tmp.cumsum().plot()
plt.title(f'SR={sr.values.flatten()}')


# Plot cumulative returns and their Sharpe Ratio
def sr_plot(predictions: np.ndarray,
            returns: np.ndarray,
            names: list = None,
            dates=None):
    # build managed returns
    data = predictions * returns.reshape(-1, 1)
    tmp = pd.DataFrame(data)
    if names is not None:
        tmp.columns = names
    if dates is not None:
        tmp.index = dates
    # tmp = pd.DataFrame(predictions)
    tmp['simple_returns'] = returns
    timed_returns = tmp / tmp.std()
    sr = sharpe_ratio(tmp)
    timed_returns.cumsum().plot()
    plt.title(f'SR={sr.values.flatten()}')