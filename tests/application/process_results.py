import warnings
import pandas as pd
import numpy as np
from pathlib import Path
import os
import scipy.stats as ss

def share_hypo(errors, baseline, significance=0.95, target_share=0.5):
    p_0 = target_share
    q_0 = 1 - p_0
    n = len(errors)
    share = np.mean(errors < baseline)
    if n * share < 5:
        return False
    if n * (1 - share) < 5:
        return True
    stat = (share - p_0) / np.sqrt(p_0 * q_0 / n)
    border = ss.norm.ppf(1 - significance)
    return stat > border


MODELS = ['ARIMA', 'ETS', 'PROPHET', 'Naive', 'DynamicRegression', 'SeasonalDynamicRegression']
cwd = Path(os.getcwd()) / 'tests' / 'application'
features = pd.read_csv(cwd / 'features.csv')
results = []
for periodicity in ['H', 'D', 'M', 'Q', 'Y']:
    period_results = pd.read_csv(cwd / f"{periodicity}_results.csv")
    if periodicity != 'Y':
        period_results['SeasonalDynamicRegression'] = period_results[[column \
                                                    for column in period_results.columns \
                                                        if column.startswith('Dynamic') and len(column.split('-')) == 3]].min(axis=1)
    period_results['DynamicRegression'] = period_results[[column \
                                                for column in period_results.columns \
                                                    if column.startswith('Dynamic') and len(column.split('-')) == 2]].min(axis=1)
    period_results = period_results.drop(
        [column for column in period_results.columns if column.startswith('Dynamic') and '-' in column],
        axis=1
    ).dropna()
    period_results = period_results.set_index('column').join(features.set_index('M4id'))
    period_results = period_results.groupby(['periodicity', 'category', 'is_seasonal', 'is_stationary', 'fh']).agg(list)
    results.append(period_results)
results = pd.concat(results)
results.to_csv(cwd / 'aggregated_results.csv')
conclusion = pd.DataFrame(index=results.index, columns=[
    'ARIMA_wins_percent',
    'ETS_wins_percent',
    'PROPHET_wins_percent',
    'Naive_wins_percent',
    'DynamicRegression_wins_percent',
    'SeasonalDynamicRegression_wins_percent',
    'best_model',
    'DynamicRegression_vs_ARIMA',
    'DynamicRegression_vs_ETS',
    'DynamicRegression_vs_PROPHET',
    'DynamicRegression_vs_Naive',
    'SeasonalDynamicRegression_vs_ARIMA',
    'SeasonalDynamicRegression_vs_ETS',
    'SeasonalDynamicRegression_vs_PROPHET',
    'SeasonalDynamicRegression_vs_Naive'
])
for index, data in results.iterrows():
    ARIMA_errors = np.array(data['ARIMA'])
    ETS_errors = np.array(data['ETS'])
    PROPHET_errors = np.array(data['PROPHET'])
    Naive_errors = np.array(data['Naive'])
    DynamicRegression_errors = np.array(data['DynamicRegression'])
    min_errors = np.min(np.vstack([ARIMA_errors, ETS_errors, PROPHET_errors, Naive_errors, DynamicRegression_errors]), axis=0)
    if index[0] != 'Y':
        SeasonalDynamicRegression_errors = np.array(data['SeasonalDynamicRegression'])
        min_errors = np.min(np.vstack([min_errors, SeasonalDynamicRegression_errors]), axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        conclusion.loc[index]['ARIMA_wins_percent'] = np.mean(min_errors == ARIMA_errors)
        conclusion.loc[index]['ETS_wins_percent'] = np.mean(min_errors == ETS_errors)
        conclusion.loc[index]['PROPHET_wins_percent'] = np.mean(min_errors == PROPHET_errors)
        conclusion.loc[index]['Naive_wins_percent'] = np.mean(min_errors == Naive_errors)
        conclusion.loc[index]['DynamicRegression_wins_percent'] = np.mean(min_errors == DynamicRegression_errors)
        conclusion.loc[index]['DynamicRegression_vs_ARIMA'] = share_hypo(DynamicRegression_errors, ARIMA_errors)
        conclusion.loc[index]['DynamicRegression_vs_ETS'] = share_hypo(DynamicRegression_errors, ETS_errors)
        conclusion.loc[index]['DynamicRegression_vs_PROPHET'] = share_hypo(DynamicRegression_errors, PROPHET_errors)
        conclusion.loc[index]['DynamicRegression_vs_Naive'] = share_hypo(DynamicRegression_errors, Naive_errors)
        conclusion.loc[index]['best_model'] = conclusion.loc[index][[
            'ARIMA_wins_percent',
            'ETS_wins_percent',
            'PROPHET_wins_percent',
            'Naive_wins_percent',
            'DynamicRegression_wins_percent',
        ]].astype(np.float64).idxmax().split('_')[0]
        if index[0] != 'Y':
            conclusion.loc[index]['SeasonalDynamicRegression_wins_percent'] = np.mean(min_errors == SeasonalDynamicRegression_errors)
            conclusion.loc[index]['SeasonalDynamicRegression_vs_ARIMA'] = share_hypo(SeasonalDynamicRegression_errors, ARIMA_errors)
            conclusion.loc[index]['SeasonalDynamicRegression_vs_ETS'] = share_hypo(SeasonalDynamicRegression_errors, ETS_errors)
            conclusion.loc[index]['SeasonalDynamicRegression_vs_PROPHET'] = share_hypo(SeasonalDynamicRegression_errors, PROPHET_errors)
            conclusion.loc[index]['SeasonalDynamicRegression_vs_Naive'] = share_hypo(SeasonalDynamicRegression_errors, Naive_errors)
            conclusion.loc[index]['best_model'] = conclusion.loc[index][[
                'ARIMA_wins_percent',
                'ETS_wins_percent',
                'PROPHET_wins_percent',
                'Naive_wins_percent',
                'DynamicRegression_wins_percent',
                'SeasonalDynamicRegression_wins_percent',
            ]].astype(np.float64).idxmax().split('_')[0]
conclusion.to_csv(cwd / 'final_results.csv')
    