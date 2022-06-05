from functools import partial
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from random import shuffle
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from pathlib import Path
    from models.dynamic_regression import DynamicRegression
    from models.models import *
    import os
    import argparse
    from statsmodels.tsa.stattools import kpss
    from statsmodels.tsa.seasonal import STL
    from tqdm import tqdm
    from multiprocessing import Pool
    from sktime.forecasting.model_selection import SlidingWindowSplitter
    from sktime.performance_metrics.forecasting import MeanAbsoluteError
    from models.models import SktimeARIMA, SktimeETS, SktimeProphet, SktimeNaive
    from models.dynamic_regression import DynamicRegression
    from sktime.forecasting.model_evaluation import evaluate
    from sklearn.metrics import mean_absolute_error


FH = {
    'H': 24,
    'D': 30,
    'W': 4,
    'M': 12,
    'Q': 4,
    'Y': 3
}

TRAIN_LENGTH = dict()

def choose_subset(info_path, size=0.1):
    info = pd.read_csv(info_path)
    if size == 1:
        return info['M4id']
    train, test = train_test_split(info, train_size=size, shuffle=True)
    return train['M4id']


def choose_subset_from_features(features, features_agg, min_cluster_size=200):
    columns = np.array([])
    features_aggregated = features_agg[features_agg['count'] >= min_cluster_size]
    progressbar = tqdm(features_aggregated.iterrows(), total=features_aggregated.shape[0])
    progressbar.set_description('Choosing columns...')
    for index, row in progressbar:
        cur_columns = features[
            (features['periodicity'] == row['periodicity']) & (features['category'] == row['category']) & \
            (features['is_seasonal'] == row['is_seasonal']) & (features['is_stationary'] == row['is_stationary'])
        ]['M4id'].to_numpy()
        np.random.shuffle(cur_columns)
        cur_columns = cur_columns[:min_cluster_size]
        columns = np.append(columns, cur_columns)
    return columns


def load_series(columns, dataset_path, periodicity):
    if periodicity == 'H':
        dataset = pd.read_csv(dataset_path / 'Hourly-train.csv')
    elif periodicity == 'D':
        dataset = pd.read_csv(dataset_path / 'Daily-train.csv')
    elif periodicity == 'W':
        dataset = pd.read_csv(dataset_path / 'Weekly-train.csv')
    elif periodicity == 'Q':
        dataset = pd.read_csv(dataset_path / 'Quarterly-train.csv')
    elif periodicity == 'M':
        dataset = pd.read_csv(dataset_path / 'Monthly-train.csv')
    elif periodicity == 'Y':
        dataset = pd.read_csv(dataset_path / 'Yearly-train.csv')
    names = columns[columns.str.startswith(periodicity)]
    output = dataset[dataset["V1"].isin(names)]
    return output

def cross_val(params, data, cv_window_cnt=5):
    try:
        model = params['model']
        column = params['column']
        periodicity = column[0]
        if periodicity == 'M':
            freq = '30D'
        elif periodicity == 'Y':
            freq = '365D'
        elif periodicity == 'Q':
            freq = '91D'
        else:
            freq = periodicity
        series = data[data["V1"] == column].squeeze().dropna().drop(index='V1')
        series = pd.Series(
            series.values,
            index=pd.DatetimeIndex(pd.date_range(end='2022-05-01 00:00', freq=freq, periods=len(series))),
            dtype=np.float64
        )
        fh = params['fh']
        sp = FH[periodicity]
        if fh == 'one-step':
            fh = [1]
        elif fh == 'long-step':
            fh = [sp]
        else:
            fh = np.arange(1, sp + 1)
        window_length = len(series) - cv_window_cnt * max(fh)
        if model == 'ARIMA' or model == 'ETS' or model == 'PROPHET' or model == 'Naive':
            if window_length < cv_window_cnt * max(fh):
                return [None, params]
        elif model.startswith('DynamicRegression'):
            ar_depth = int(model.split('-')[1])
            if window_length < ar_depth + max(fh):
                return [None, params]
        else:
            if periodicity == 'Y':
                return [None, params]
            ar_depth, seas_depth = map(int, model.split('-')[1:])
            if window_length < max(ar_depth, seas_depth * sp) + max(fh):
                return [None, params]
        cv = SlidingWindowSplitter(fh=fh, window_length=window_length, step_length = int(max(fh)), start_with_window=True)
        if model == 'ARIMA':
            model = SktimeARIMA(suppress_warnings=True)
        elif model == 'ETS':
            model = SktimeETS(auto=True, n_jobs=-1)
        elif model == 'PROPHET':
            model = SktimeProphet()
        elif model == 'Naive':
            model = SktimeNaive()
        elif model.startswith('DynamicRegression'):
            ar_depth = int(model.split('-')[1])
            model = DynamicRegression(fh=fh, ar_depth=ar_depth)
        else:
            ar_depth, seas_depth = map(int, model.split('-')[1:])
            model = DynamicRegression(fh=fh, sp=sp, ar_depth=ar_depth, seas_depth=seas_depth)
        errors = []
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for train_inds, test_inds in cv.split(series):
                train = series.iloc[train_inds]
                test = series.iloc[test_inds]
                model.fit(train)
                if isinstance(model, DynamicRegression):
                    predicts = model.predict(train)
                elif isinstance(model, SktimeProphet):
                    predicts = model.predict(fh=fh).to_numpy().squeeze()
                else:
                    predicts = model.predict(fh=fh).squeeze()
                errors.append(mean_absolute_error([test], [predicts]))
        mae = np.mean(errors)
        return [mae, params]
    except:
        print("EXCEPTION IN APPLY MODELS")
        print(params)
        return [None, params]


def apply_models(df, data, cv_window_cnt=5):
    grid = []
    for column, fh in df.index:
        for model in df.columns:
            grid.append({'column': column, 'fh': fh, 'model': model})
    cross_val_function = partial(
        cross_val,
        data=data,
        cv_window_cnt=cv_window_cnt
    )
    with Pool(processes=os.cpu_count() - 1) as pool:
        progress_bar = tqdm(pool.imap_unordered(cross_val_function, grid), total=len(grid), dynamic_ncols=True)
        progress_bar.set_description(f"Calculating periodicity {grid[0]['column'][0]}...")
        returns = list(progress_bar)
    progress_bar = tqdm(returns, dynamic_ncols=True, leave=False)
    progress_bar.set_description("Writing into dataframe...")
    for mae, params in returns:
        df.loc[(params['column'], params['fh'])][params['model']] = mae
    return df


def calculate_series_features(df, info_path, stationary_threshold_pvalue=0.05, seasonal_threshold=0.2):
    info = pd.read_csv(info_path)
    df = df.sort_index()
    periodicity = None
    dataset = None
    progressbar = tqdm(df.index)
    progressbar.set_description("Calculating features")
    for column in progressbar:
        cur_periodicity = column[0]
        if cur_periodicity != periodicity:
            periodicity = cur_periodicity
            if periodicity == 'H':
                dataset = pd.read_csv(dataset_path / 'Hourly-train.csv')
            elif periodicity == 'D':
                dataset = pd.read_csv(dataset_path / 'Daily-train.csv')
            elif periodicity == 'W':
                dataset = pd.read_csv(dataset_path / 'Weekly-train.csv')
            elif periodicity == 'Q':
                dataset = pd.read_csv(dataset_path / 'Quarterly-train.csv')
            elif periodicity == 'M':
                dataset = pd.read_csv(dataset_path / 'Monthly-train.csv')
            elif periodicity == 'Y':
                dataset = pd.read_csv(dataset_path / 'Yearly-train.csv')
        cur_series = dataset[dataset["V1"] == column].squeeze().dropna().drop(index='V1')
        stl = STL(cur_series, period=4)
        res = stl.fit()
        seas_strength = max(0, 1 - res.resid.std()**2 / (res.resid.std()**2 + res.seasonal.std()**2))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            kpss_results = kpss(cur_series - res.seasonal, nlags='auto')
        is_seasonal = seas_strength > seasonal_threshold
        is_stationary = kpss_results[1] > stationary_threshold_pvalue
        df.loc[column]['periodicity'] = periodicity
        df.loc[column]['category'] = info[info['M4id'] == column]['category'].values[0]
        df.loc[column]['is_seasonal'] = is_seasonal
        df.loc[column]['is_stationary'] = is_stationary
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-ar-depth", type=int)
    parser.add_argument("--max-seas-depth", type=int)
    parser.add_argument("--cv-windows", type=int)
    args = parser.parse_args()
    if not args.max_ar_depth or not args.max_seas_depth or not args.cv_windows:
        raise TypeError()
    m4_path = Path((Path(os.getcwd()) / 'Dataset'))
    dataset_path = m4_path / 'Train'
    info_path = m4_path / 'M4-info.csv'

    full_columns = choose_subset(info_path, 1).sort_values()
    features = pd.DataFrame(index=full_columns, columns=['periodicity', 'category', 'is_seasonal', 'is_stationary'])
    features = calculate_series_features(features, info_path)
    features.to_csv(Path(os.getcwd()) / 'tests' / 'application' / 'features.csv')
    features['count'] = 1
    features_aggregated = features.groupby(by=['periodicity', 'category', 'is_seasonal', 'is_stationary']).sum()
    features_aggregated.to_csv(Path(os.getcwd()) / 'tests' / 'application' / 'features_aggregated.csv')
    features = pd.read_csv((Path(os.getcwd()) / 'tests' / 'application' / 'features.csv'))
    features_aggregated = pd.read_csv((Path(os.getcwd()) / 'tests' / 'application' / 'features_aggregated.csv'))

    columns = pd.Series(choose_subset_from_features(features, features_aggregated)).sort_values().reset_index(drop=True)
    columns.to_csv(Path(os.getcwd()) / 'tests' / 'application' / 'columns.csv')
    columns = pd.read_csv(Path(os.getcwd()) / 'tests' / 'application' / 'columns.csv')['0']

    models = [
        'ARIMA',
        'ETS',
        'PROPHET',
        'Naive',
        *[f"DynamicRegression-{ar_depth}" for ar_depth in range(1, int(args.max_ar_depth) + 1)],
        *[f"DynamicRegression-{ar_depth}-{seas_depth}" \
            for ar_depth in range(1, int(args.max_ar_depth) + 1) \
                for seas_depth in range(1, int(args.max_seas_depth) + 1)]
    ]
    for periodicity in ['H', 'D', 'M', 'Q', 'Y']:
        indices = []
        for column in columns[columns.str.startswith(periodicity)]:
            indices.append((column, 'one-step'))
            indices.append((column, 'long-step'))
            indices.append((column, 'many-steps'))
        models_results = pd.DataFrame(index=pd.MultiIndex.from_tuples(indices, names=['column', 'fh']), columns=models)
        dataset = load_series(columns, dataset_path, periodicity)
        dataset.to_csv(Path(os.getcwd()) / 'tests' / 'application' / f"{periodicity}.csv")
        models_results = apply_models(models_results, dataset, args.cv_windows)
        models_results.to_csv(Path(os.getcwd()) / 'tests' / 'application' / f'{periodicity}_results.csv')
