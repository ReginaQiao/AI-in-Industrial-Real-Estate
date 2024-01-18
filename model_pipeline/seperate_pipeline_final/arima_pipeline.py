import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import pmdarima as pm
from pmdarima.arima.stationarity import ADFTest
from pmdarima.arima import ndiffs
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

import multiprocessing as mp
from datetime import datetime as dtm
from typing import Optional, Sequence
import itertools
from sklearn.model_selection import ParameterGrid as PG
import joblib
from sklearn.feature_selection import mutual_info_regression as MIR
import time
from sklearn.model_selection import TimeSeriesSplit
from pmdarima.arima import auto_arima


def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    return np.mean(numerator / denominator) * 100


def get_important_features(df, feature_space, k, thresh):
    y = df['real_hedonic_rent_submarket']
    X = df[feature_space]

    mi_scores = MIR(X, y)
    # print(mi_scores)

    features = []
    if thresh:
        mi_score_selected_index = np.where(mi_scores > thresh)[0]
        features = X.columns[mi_score_selected_index].tolist()
        # print(f"num features above mi thresh for submarket {name}: {len(features)}")

    if k:
        mi_score_selected_index = np.argsort(mi_scores)[::-1][:k]
        features = X.columns[mi_score_selected_index].tolist()

    return features


def _get_list_intersect_size(list1, list2):
    return len(set(list1) & set(list2))


def get_feature_subsets(
        feature_space,
        subset_size,
        include_features=None,
        intersect_size=1,
):
    subset_size = max(1, subset_size)
    subset_size = min(subset_size, len(feature_space))
    subset_li = []

    for k in range(subset_size, 0, -1):
        for subset in itertools.combinations(feature_space, k):
            subset = list(subset)
            if include_features is not None:
                intersect_size = min(k, intersect_size)
                _get_list_intersect_size(subset, include_features)

            subset_li.append(subset)

    return subset_li


def _process_subset(subset):
    return subset


def run_auto_arima_experiment(name, exo, group, params, ntest, min_p, max_p, min_q, max_q, diff):
    Y_train = group['real_hedonic_rent_submarket'][:-ntest]
    Y_test = group['real_hedonic_rent_submarket'][-ntest:]
    X_train = exo[params['subset_li']].iloc[:-ntest, :]
    X_test = exo[params['subset_li']].iloc[-ntest:, :]

    adf_test = ADFTest(alpha=0.05)
    p_val, should_diff = adf_test.should_diff(group['real_hedonic_rent_submarket'])
    kpss_diffs = ndiffs(group['real_hedonic_rent_submarket'], alpha=0.05, test='kpss', max_d=6)
    adf_diffs = ndiffs(group['real_hedonic_rent_submarket'], alpha=0.05, test='adf', max_d=6)
    n_diffs = max(adf_diffs, kpss_diffs)
    print(name, ': ', p_val, should_diff, n_diffs)

    if diff:
        auto = pm.auto_arima(Y_train, X_train, d=n_diffs,
                             suppress_warnings=True, error_action="ignore",
                             min_p=min_p, min_q=min_q, max_p=max_p, max_q=max_q,
                             stepwise=True, scoring=smape,
                             max_order=None, trace=True)
    else:
        auto = pm.auto_arima(Y_train, X_train, d=0,
                             suppress_warnings=True, error_action="ignore",
                             min_p=min_p, min_q=min_q, max_p=max_p, max_q=max_q,
                             stepwise=True, scoring=smape,
                             max_order=None, trace=True)

    model = auto
    y_train_pred = model.predict(len(Y_train), X_train)
    y_test_pred = model.predict(ntest, X_test)
    p, d, q = model.order

    mse_train = mean_squared_error(Y_train, y_train_pred)
    mse_test = mean_squared_error(Y_test, y_test_pred)
    smape_train = smape(Y_train, y_train_pred)
    smape_test = smape(Y_test, y_test_pred)

    n_params = model.order[0] + model.order[1] + model.order[2] + len(params['subset_li'])
    n_obs = len(Y_train)
    aic = n_obs * np.log(mse_train) + 2 * n_params
    bic = n_obs * np.log(mse_train) + n_params * np.log(n_obs)

    model_info = pd.DataFrame({
        "research_submkt_id": name,
        "Date": group['date'],
        "Y": pd.concat([Y_train, Y_test], axis=0).tolist(),
        "Y_pred": pd.concat([y_train_pred, y_test_pred], axis=0).tolist(),
        "Train_SMAPE": pd.Series(
            [smape_train for i in range(len(Y_train))] + [smape_test for i in range(ntest)]).tolist(),
        "Train_MSE": pd.Series([mse_train for i in range(len(Y_train))] + [mse_test for i in range(ntest)]).tolist(),
        "p": p,
        "d": d,
        "q": q
    })

    return model_info


def run_auto_arima_pipeline(df, ntest, feature_space, k, thresh, min_p, max_p, min_q, max_q, test_date, diff):
    results = {}
    best_models_df = pd.DataFrame()
    best_df = pd.DataFrame()

    grouped = df.groupby('research_submkt_id')
    for name, group in grouped:
        features = get_important_features(group, feature_space, k, thresh)
        exo = group[features]
        subset_li = get_feature_subsets(
            features,
            subset_size=len(features),
            include_features=None,
            intersect_size=1
        )

        param_vals = {
            "subset_li": subset_li
        }
        param_grid = list(PG(param_vals))
        num_params = len(param_grid)

        pool = mp.Pool(processes=mp.cpu_count())
        for idx, params in enumerate(param_grid):
            print(f"training model {idx}/{num_params - 1}: {params}")

            result = pool.apply_async(
                run_auto_arima_experiment,
                kwds={
                    "name": name,
                    "exo": exo,
                    "group": group,
                    "params": params,
                    "ntest": ntest,
                    "min_p": min_p,
                    "max_p": max_p,
                    "min_q": min_q,
                    "max_q": max_q,
                    "diff": diff
                },
            )
            params = str(params)
            results[params] = result

        pool.close()
        pool.join()

        best_submkt_whole_df = pd.DataFrame()
        best_submkt_part_df = pd.DataFrame()

        for key, value in results.items():
            dic = value.get()
            date = dic['Date']
            y_true = dic['Y']
            y_pred = dic['Y_pred']
            train_smape = dic['Train_SMAPE']
            train_mse = dic['Train_MSE']

            data = {
                'research_submkt_id': name,
                'date': date,
                'y_true': y_true,
                'y_pred': y_pred,
                'smape': train_smape,
                'mse': train_mse,
                'attributes': key
            }
            rs_part_df = pd.DataFrame(data)
            smape_choose = float('inf')
            if rs_part_df[rs_part_df['date'] == test_date]['smape'].iloc[0] < smape_choose:
                best_submkt_whole_df = rs_part_df
                best_submkt_part_df = rs_part_df[rs_part_df['date'] >= test_date]

        best_models_df = pd.concat([best_models_df, best_submkt_whole_df])
        best_df = pd.concat([best_df, best_submkt_part_df])

    return best_models_df, best_df




def run_auto_arima_experiment_cv(name, exo, group, params, ntest, min_p, max_p, min_q, max_q, diff):
    adf_test = ADFTest(alpha=0.05)
    p_val, should_diff = adf_test.should_diff(group['real_hedonic_rent_submarket'])
    kpss_diffs = ndiffs(group['real_hedonic_rent_submarket'], alpha=0.05, test='kpss', max_d=6)
    adf_diffs = ndiffs(group['real_hedonic_rent_submarket'], alpha=0.05, test='adf', max_d=6)
    n_diffs = max(adf_diffs, kpss_diffs)
    print(name, ': ', p_val, should_diff, n_diffs)

    Y_train = group['real_hedonic_rent_submarket'][:-ntest]
    Y_test = group['real_hedonic_rent_submarket'][-ntest:]
    X_train = exo[params['subset_li']].iloc[:-ntest, :]
    X_test = exo[params['subset_li']].iloc[-ntest:, :]

    # Define the number of cross-validation folds: 5
    n_folds = 5
    tscv = TimeSeriesSplit(n_splits=n_folds)

    mse_train_scores = []
    arima_models = []

    for train_index, val_index in tscv.split(Y_train):
        Y_train_fold, Y_val_fold = Y_train.iloc[train_index], Y_train.iloc[val_index]
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]

        adf_test = ADFTest(alpha=0.05)
        p_val, should_diff = adf_test.should_diff(Y_train_fold)
        kpss_diffs = ndiffs(Y_train_fold, alpha=0.05, test='kpss', max_d=6)
        adf_diffs = ndiffs(Y_train_fold, alpha=0.05, test='adf', max_d=6)
        n_diffs = max(adf_diffs, kpss_diffs)

        if diff:
            auto = auto_arima(Y_train_fold, X_train_fold, d=n_diffs,
                              suppress_warnings=True, error_action="ignore",
                              min_p=min_p, min_q=min_q, max_p=max_p, max_q=max_q,
                              stepwise=True, scoring=smape, max_order=None, trace=False)
        else:
            auto = auto_arima(Y_train_fold, X_train_fold, d=0,
                              suppress_warnings=True, error_action="ignore",
                              min_p=min_p, min_q=min_q, max_p=max_p, max_q=max_q,
                              stepwise=True, scoring=smape, max_order=None, trace=False)

        model = auto
        arima_models.append(model)

        y_val_pred = model.predict(len(Y_val_fold), X_val_fold)

        mse_val_pred = mean_squared_error(Y_val_fold, y_val_pred)
        mse_train_scores.append(mse_val_pred)

    # Find the model with the smallest MSE
    best_model_idx = np.argmin(mse_train_scores)
    best_model = arima_models[best_model_idx]

    # Fit the best model on the full training data
    best_model.fit(Y_train, exogenous=X_train)

    p, d, q = best_model.order

    # Make predictions on the train and test sets
    y_train_pred = best_model.predict(n_periods=len(Y_train), exogenous=X_train)
    y_test_pred = best_model.predict(n_periods=ntest, exogenous=X_test)

    mse_train = mean_squared_error(Y_train, y_train_pred)
    mse_test = mean_squared_error(Y_test, y_test_pred)
    smape_train = smape(Y_train, y_train_pred)
    smape_test = smape(Y_test, y_test_pred)

    n_params = model.order[0] + model.order[1] + model.order[2] + len(params['subset_li'])
    n_obs = len(Y_train)
    aic = n_obs * np.log(mse_train) + 2 * n_params
    bic = n_obs * np.log(mse_train) + n_params * np.log(n_obs)

    model_info = pd.DataFrame({
        "research_submkt_id": name,
        "date": group['date'],
        "Y": pd.concat([Y_train, Y_test], axis=0).tolist(),
        "Y_pred": pd.concat([y_train_pred, y_test_pred], axis=0).tolist(),
        "Train_SMAPE": pd.Series(
            [smape_train for i in range(len(Y_train))] + [smape_test for i in range(ntest)]).tolist(),
        "Train_MSE": pd.Series([mse_train for i in range(len(Y_train))] + [mse_test for i in range(ntest)]).tolist(),
        "p": p,
        "d": d,
        "q": q
    })

    return model_info


def run_auto_arima_pipeline_cv(df, ntest, feature_space, k, thresh, min_p, max_p, min_q, max_q, test_date, diff):
    results = {}
    best_models_df = pd.DataFrame()
    best_df = pd.DataFrame()

    grouped = df.groupby('research_submkt_id')
    for name, group in grouped:
        features = get_important_features(group, feature_space, k, thresh)
        exo = group[features]
        subset_li = get_feature_subsets(
            features,
            subset_size=len(features),
            include_features=None,
            intersect_size=1
        )

        param_vals = {
            "subset_li": subset_li
        }
        param_grid = list(PG(param_vals))
        num_params = len(param_grid)

        pool = mp.Pool(processes=mp.cpu_count())
        for idx, params in enumerate(param_grid):
            print(f"training model {idx}/{num_params - 1}: {params}")

            result = pool.apply_async(
                run_auto_arima_experiment_cv,
                kwds={
                    "name": name,
                    "exo": exo,
                    "group": group,
                    "params": params,
                    "ntest": ntest,
                    "min_p": min_p,
                    "max_p": max_p,
                    "min_q": min_q,
                    "max_q": max_q,
                    "diff": diff
                },
            )
            params = str(params)
            results[params] = result

        pool.close()
        pool.join()

        best_submkt_whole_df = pd.DataFrame()
        best_submkt_part_df = pd.DataFrame()

        for key, value in results.items():
            dic = value.get()
            date = dic['date']
            y_true = dic['Y']
            y_pred = dic['Y_pred']
            train_smape = dic['Train_SMAPE']
            train_mse = dic['Train_MSE']

            data = {
                'research_submkt_id': name,
                'date': date,
                'y_true': y_true,
                'y_pred': y_pred,
                'smape': train_smape,
                'mse': train_mse,
                'attributes': key
            }
            rs_part_df = pd.DataFrame(data)
            smape_choose = float('inf')
            if rs_part_df[rs_part_df['date'] == test_date]['smape'].iloc[0] < smape_choose:
                best_submkt_whole_df = rs_part_df
                best_submkt_part_df = rs_part_df[rs_part_df['date'] >= test_date]

        best_models_df = pd.concat([best_models_df, best_submkt_whole_df])
        best_df = pd.concat([best_df, best_submkt_part_df])

    return best_models_df, best_df


if __name__ == "__main__":

    feature_space = [
       'population_histfc','nominal_retail_sales_histfc',
       'nominal_earnings_by_residence_histfc',
       'gdp_histfc', 'unemployment_histfc', 'employment_histfc',
       'unemployment_rate_histfc', 'labor_force_histfc',
       'manufacturing_employment_histfc', 'employment_trade_histfc',
       'employment_warehousing_histfc', 'affordability_index_histfc',
       'gdp_transp_and_dist_histfc', 'income_per_capita_histfc',
       'median_sfh_sale_price_histfc', 'tech_employment_histfc',
       'household_count_histfc', 'nominal_earnings_by_workplace_histfc',
       'nominal_proprietors_income_histfc', 'housing_completions_histfc',
       'employment_wholesale_trade_histfc', 'population_20_24_histfc',
       'population_25_29_histfc', 'population_30_34_histfc',
       'population_35_39_histfc', 'population_40_44_histfc',
       'population_45_49_histfc', 'real_retail_sales_histfc',
       'real_earnings_by_residence_histfc',
       'real_earnings_by_workplace_histfc', 'real_proprietors_income_histfc',
       'ecomm_sh', 'real_retail_sales_ex_gas',
       'real_bricks_and_mortar_retail_sales', 'real_ecommerce',
       'ecomm_footprint_adj_sales', 'exports_us', 'imports_us', 'treasury_10y',
       'spread_3m10y', 'sofr_3m', 'baa_credit_spreads', 'cpi_inflation',
       'cpi_trailing_12qtr_cagr','real_market_level_rent','ecomm_pop',
       'ecomm^2_pop', 'weighted_pop_estimate_cryr',
       'weighted_hh_estimate_cryr','total_dock_doors', 'total_car_spaces', 'retailirsa', 'pcedg', 'pcend',
       'pcepilfe', 'rsxfs', 'isratio', 'mrtsir4423xuss', 'whlslrimsa',
       'a333rx1q020sbea']

    df_dal = pd.read_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/dal_submkt_train_test_data.csv',
        index_col=0)

    df_pho = pd.read_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/pho_submkt_train_test_data.csv',
        index_col=0)

    min_p = 1
    max_p = 24
    min_q = 1
    max_q = 24
    test_date = '2021-08-01'

    # without cv
    start_time = time.time()

    dal_24_8_df1, dal_24_8_df2 = run_auto_arima_pipeline(df_dal, 24, feature_space, 8, None, min_p, max_p, min_q, max_q,
                                                         test_date, True)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")

    dal_24_8_df1.to_csv('/mnt/container1/np_forecast_data/zqiao_data/arima_result/DAL/dal_24_8_train_test.csv')
    dal_24_8_df2.to_csv('/mnt/container1/np_forecast_data/zqiao_data/arima_result/DAL/dal_24_8_test.csv')

    # without cv and d=0
    start_time = time.time()
    dal_24_8_df1_d0, dal_24_8_df2_d0 = run_auto_arima_pipeline(df_dal, 24, feature_space, 8, None, min_p, max_p, min_q,
                                                               max_q, test_date, False)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")

    dal_24_8_df1_d0.to_csv('/mnt/container1/np_forecast_data/zqiao_data/arima_result/DAL/dal_24_8_d0_train_test.csv')
    dal_24_8_df2_d0.to_csv('/mnt/container1/np_forecast_data/zqiao_data/arima_result/DAL/dal_24_8_d0_test.csv')

