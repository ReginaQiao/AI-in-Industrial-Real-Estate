import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
import pmdarima as pm
from pmdarima.arima.stationarity import ADFTest
from pmdarima.arima import ndiffs
from sklearn.metrics import mean_squared_error
from pmdarima.metrics import smape
from statsmodels.tsa.arima.model import ARIMA

import multiprocessing as mp
from datetime import datetime as dtm
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


def smape_scorer(y_true, y_pred):
    return -smape(y_true, y_pred)


def get_important_features(df, feature_space, k, thresh):
    y = df['real_hedonic_rent_submarket']
    X = df[feature_space]

    mi_scores = MIR(X, y)

    features = []
    if thresh:
        mi_score_selected_index = np.where(mi_scores > thresh)[0]
        features = X.columns[mi_score_selected_index].tolist()

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
                             stepwise=True, scoring=smape_scorer,
                             max_order=None, trace=True)
    else:
        auto = pm.auto_arima(Y_train, X_train, d=0,
                             suppress_warnings=True, error_action="ignore",
                             min_p=min_p, min_q=min_q, max_p=max_p, max_q=max_q,
                             stepwise=True, scoring=smape_scorer,
                             max_order=None, trace=True)

    model = auto
    y_pred = model.predict(ntest, X_test)
    p, d, q = model.order

    date = group['date'].iloc[-ntest:].tolist()
    mse_test = mean_squared_error(Y_test, y_pred)
    smape_test = smape(Y_test, y_pred)

    model_info = pd.DataFrame({
        "research_submkt_id": [name] * ntest,  # Repeat the name ntest times to match the length
        "date": date,
        "y_test": Y_test.values,
        "y_pred": y_pred,
        "mse": [mse_test] * ntest,  # Repeat the mse value ntest times to match the length
        "smape": [smape_test] * ntest,  # Repeat the smape value ntest times to match the length
        "p": [p] * ntest,  # Repeat the p value ntest times to match the length
        "d": [d] * ntest,  # Repeat the d value ntest times to match the length
        "q": [q] * ntest,  # Repeat the q value ntest times to match the length
        "best_attributes": [params['subset_li']] * ntest
    })

    return model_info


def run_auto_arima_pipeline(df, ntest, feature_space, k, thresh, min_p, max_p, min_q, max_q, diff):
    rs_df = pd.DataFrame()
    best_models_df = pd.DataFrame()

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

        results = {}
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

        for key, value in results.items():
            dic = value.get()
            date = dic['date']
            y_test = dic['y_test']
            y_pred = dic['y_pred']
            smape_test = dic['smape']
            mse_test = dic['mse']

            data = {
                'research_submkt_id': name,
                'date': date,
                'y_test': y_test,
                'y_pred': y_pred,
                'smape': smape_test,
                'mse': mse_test,
                'attributes': key
            }
            rs_part_df = pd.DataFrame(data)

            rs_df = pd.concat([rs_df, rs_part_df])

        best_models_df = pd.concat([best_models_df, rs_df])

    grouped_ = best_models_df.groupby('research_submkt_id')

    rs_df_new = pd.DataFrame()

    for name, group_ in grouped_:
        grdf = group_[group_['smape'] == group_['smape'].min()]
        rs_df_new = pd.concat([rs_df_new, grdf])

    return rs_df_new


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
       'spread_3m10y', 'sofr_3m', 'real_market_level_rent','ecomm_pop',
       'ecomm^2_pop', 'weighted_pop_estimate_cryr',
       'weighted_hh_estimate_cryr','total_dock_doors', 'total_car_spaces',
       'retailirsa', 'pcedg', 'pcend', 'pcepilfe', 'rsxfs', 'isratio',
       'mrtsir4423xuss', 'whlslrimsa','a333rx1q020sbea','n_properties_10,000 sf to 30,000 sf',
        'total_property_sqft_10,000 sf to 30,000 sf','n_properties_175,000 sf to 400,000 sf',
        'total_property_sqft_175,000 sf to 400,000 sf','n_properties_30,000 sf to 75,000 sf',
        'total_property_sqft_30,000 sf to 75,000 sf','n_properties_400,000 sf to 800,000 sf',
        'total_property_sqft_400,000 sf to 800,000 sf','n_properties_75,000 sf to 175,000 sf',
        'total_property_sqft_75,000 sf to 175,000 sf','n_properties_800,000 sf+',
        'total_property_sqft_800,000 sf+', 'n_properties_< 10,000 sf','total_property_sqft_< 10,000 sf',
        'n_properties','total_property_sqft']

    df_dal = pd.read_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/dal_submkt_train_test_data.csv',
        index_col=0)

    df_pho = pd.read_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/pho_submkt_train_test_data.csv',
        index_col=0)

    min_p = 1
    max_p = 36
    min_q = 1
    max_q = 36

    # without cv
    start_time = time.time()

    dal_36_8_ol = run_auto_arima_pipeline(df_dal, 24, feature_space, 8, None, min_p, max_p, min_q, max_q,
                                                         True)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")

    dal_36_8_ol.to_csv('/mnt/container1/np_forecast_data/zqiao_data/arima_result/DAL/dal_36_8_ol_sfnew.csv')


    start_time = time.time()

    pho_36_8_ol = run_auto_arima_pipeline(df_pho, 24, feature_space, 8, None, min_p, max_p, min_q, max_q,
                                                         True)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")

    pho_36_8_ol.to_csv('/mnt/container1/np_forecast_data/zqiao_data/arima_result/PHO/pho_36_8_ol_sfnew.csv')
