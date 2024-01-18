import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import multiprocessing as mp
from typing import List, Optional
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor
from sklearn.model_selection import ParameterGrid as PG
import time
from sklearn.feature_selection import mutual_info_regression as MIR


def get_important_features(df, feature_space, k, thresh):
    submarket_features_dict = {}
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


def ntest_split_data_by_submarket(data, ntest):
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    submarkets = data['research_submkt_id'].unique()

    for submarket in submarkets:
        submarket_data = data[data['research_submkt_id'] == submarket]
        train_submarket = submarket_data.iloc[:-ntest]
        test_submarket = submarket_data.iloc[-ntest:]
        train_data = pd.concat([train_data, train_submarket])
        test_data = pd.concat([test_data, test_submarket])

    return train_data, test_data


def smape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    return np.mean(numerator / denominator) * 100


def all_get_submkt_forecast(df, num_lags, ntest, feature_subset):
    X = df.iloc[:, [0, 1] + list(range(3, len(df.columns)))]
    Y = df.iloc[:, :3]

    Y_train, Y_test = ntest_split_data_by_submarket(Y, ntest)
    y_train = Y_train.iloc[:, -1]
    y_test = Y_test.iloc[:, -1]
    X_train, X_test = ntest_split_data_by_submarket(X, ntest)
    x_train = X_train.iloc[:, 2:]
    x_test = X_test.iloc[:, 2:]

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.1, 0.01, 0.001]
    }

    model = XGBRegressor()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(x_train, y_train)

    # Get the best parameters and best score
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    print("Best Parameters:", best_params)
    print("Best Score (RMSE):", best_score)

    best_model = XGBRegressor(**best_params)
    best_model.fit(x_train, y_train)

    importance_scores = best_model.feature_importances_
    attribute_names = x_train.columns
    attribute_importance_dict = dict(zip(attribute_names, importance_scores))

    y_pred = best_model.predict(x_test)

    Y_test_pred = Y_test.copy()
    Y_test_pred['y_pred'] = y_pred

    whole_smape = smape(y_test, y_pred)

    smape_dic = {}
    submkt_id = df['research_submkt_id'].unique().tolist()
    for submkt in submkt_id:
        y_submkt_test = Y_test_pred[Y_test_pred['research_submkt_id'] == submkt]['real_hedonic_rent_submarket']
        y_submkt_pred = Y_test_pred[Y_test_pred['research_submkt_id'] == submkt]['y_pred']
        submkt_smape = smape(y_submkt_test, y_submkt_pred)
        smape_dic[submkt] = submkt_smape

    # print(f"{submkt} - SMAPE: {smape_value:.2f}%")

    return {
        'Y_test_pred': Y_test_pred,
        'smape_dic': smape_dic,
        'whole_smape': whole_smape,
        'best_params': best_params,
        'attribute_importance': attribute_importance_dict
    }


def get_feature_subsets(
        feature_space,
        subset_size,
        include_features: Optional[List[str]] = None,
        intersect_size: int = 1,
):
    subset_size = max(1, subset_size)
    subset_size = min(subset_size, len(feature_space))
    subset_li = []

    for subset in itertools.combinations(feature_space, subset_size):
        subset = list(subset)
        if include_features is not None:
            intersect_size = min(subset_size, intersect_size)
        subset_li.append(subset)

    return subset_li


def run_attribute_permutation_pipeline(df, num_lags, ntest, feature_space, subset_size, intersect_size, top_features):
    df_new = df[['date', 'research_submkt_id', 'real_hedonic_rent_submarket']]
    df_new['date'] = pd.to_datetime(df_new['date'])
    df['date'] = pd.to_datetime(df['date'])

    if num_lags is None:
        num_lags = 36

    num_lags = num_lags
    for lag in range(1, num_lags + 1):
        df_new['rent_{}months_ago'.format(lag)] = df_new.groupby('research_submkt_id')[
            'real_hedonic_rent_submarket'].shift(lag)
    df_new = df_new.dropna()
    df_new = df_new.sort_values(['date', 'research_submkt_id']).reset_index(drop=True)

    df_sel = df[['date', 'research_submkt_id'] + feature_space]
    df_new = df_new.merge(df_sel, on=['date', 'research_submkt_id'], how='left')
    feature_space_ = df_new.columns.tolist()[3:]
    features = get_important_features(df_new, feature_space_, top_features, None)
    feature_subsets = get_feature_subsets(
        features,
        subset_size=subset_size,
        include_features=None,
        intersect_size=intersect_size
    )

    param_vals = {
        "feature_subsets": feature_subsets
    }
    param_grid = list(PG(param_vals))
    num_params = len(param_grid)

    results = {}
    pool = mp.Pool(processes=mp.cpu_count())

    for idx, params in enumerate(param_grid):
        print(f"training model {idx}/{num_params - 1}: {params}")
        result = pool.apply_async(all_get_submkt_forecast, kwds={
            "df": df_new,
            "num_lags": num_lags,
            "ntest": ntest,
            "feature_subset": params['feature_subsets'],

        })
        params = str(params)
        results[params] = result

    pool.close()
    pool.join()

    df_parts = []

    # Iterate over each key-value pair in 'results'
    for key, value in results.items():
        dic = value.get()
        y_pred_df = dic['Y_test_pred']
        smape_dic = dic['smape_dic']
        best_params = dic['best_params']
        attribute_importance = dic['attribute_importance']
        data = []

        for index, row in y_pred_df.iterrows():
            research_submkt_id = row['research_submkt_id']
            date = row['date']
            real_hedonic_rent_submarket = row['real_hedonic_rent_submarket']
            y_pred = row['y_pred']

            smape = smape_dic.get(research_submkt_id)

            submarket_best_params = best_params

            submarket_attribute_importance = attribute_importance

            submarket_info = {
                'research_submkt_id': research_submkt_id,
                'date': date,
                'real_hedonic_rent_submarket': real_hedonic_rent_submarket,
                'y_pred': y_pred,
                'smape': smape,
                'best_hyperparams': submarket_best_params,
                'best_attributes': key,
                'attribute_importance': submarket_attribute_importance
            }

            data.append(submarket_info)

        df_part = pd.DataFrame(data)

        df_parts.append(df_part)

    df = pd.concat(df_parts, ignore_index=True)
    min_smape_index = df.groupby('research_submkt_id')['smape'].idxmin()
    smallest_smape_df = df.loc[
        min_smape_index, ['research_submkt_id', 'smape', 'best_hyperparams', 'best_attributes', 'attribute_importance']]

    return df, smallest_smape_df


def plot_submkt_forecast(df, smallest_smape_df):
    for index, row in smallest_smape_df.iterrows():
        research_submkt_id = row['research_submkt_id']
        group = row['best_attributes']
        smape = row['smape']

        # Filter 'df' based on 'research_submkt_id' and 'group'-best_attributes
        submarket_df = df[(df['research_submkt_id'] == research_submkt_id) & (df['best_attributes'] == group)]

        # Extract the test and predicted values
        dates = submarket_df['date']
        y_test = submarket_df['real_hedonic_rent_submarket']
        y_pred = submarket_df['y_pred']

        # Plot the test and predicted values
        plt.plot(dates, y_test, label='Test')
        plt.plot(dates, y_pred, label='Predicted')
        # plt.yticks([4.5,4.75,5.0,5.25,5.5,5.75,6.0], ['4.5','4.75','5.0','5.25','5.5','5.75','6.0'])
        plt.xlabel('Date')
        plt.ylabel('Rent')
        plt.title(
            f'Submarket {research_submkt_id} - Group {group} (SMAPE: {smape:.2f})')  # Include SMAPE in the plot title
        plt.legend()

        # Add SMAPE as a text annotation
        plt.annotate(f'SMAPE: {smape:.2f}', xy=(0.02, 0.92), xycoords='axes fraction')

        plt.show()


if __name__ == "__main__":
    # Phoenix
    df_pho = pd.read_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/pho_submkt_train_test_data.csv')
    num_lags_pho_12 = 12
    ntest_pho_24 = 24
    subset_size_pho_24 = 24
    intersect_size_pho_1 = 1
    top_features_pho_25 = 25
    feature_sel = [
        'total_property_sqft',
        'gdp_histfc',
        'population_histfc',
        'nominal_retail_sales_histfc',
        'employment_warehousing_histfc',
        'unemployment_rate_histfc',
        'income_per_capita_histfc',
        'real_bricks_and_mortar_retail_sales',
        'nominal_proprietors_income_histfc',
        'housing_completions_histfc',
        'real_ecommerce',
        'imports_us',
        'spread_3m10y',
        'sofr_3m',
        'cpi_trailing_12qtr_cagr',
        'ecomm^2_pop',
        'real_market_level_rent',
        'weighted_pop_estimate_cryr',
        'weighted_hh_estimate_cryr',
        'total_dock_doors',
        'total_car_spaces',
        'dock_door_ratio',
        'number_of_car_spaces_ratio',
        'retailirsa',
        'pcedg',
        'pcend',
        'pcepilfe',
        'rsxfs',
        'isratio',
        'mrtsir4423xuss',
        'whlslrimsa',
        'a333rx1q020sbea']
    start_time = time.time()

    df_pho1, smallest_smape_df_pho1 = run_attribute_permutation_pipeline(df_pho, num_lags_pho_12, ntest_pho_24,
                                                                         feature_sel, subset_size_pho_24,
                                                                         intersect_size_pho_1, top_features_pho_25)

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")

    df_pho1.to_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_forecsat_model/seperate_pipeline_final/xgboost_result_flake/xgboost_fs/pho_24_12_pred.csv')
    smallest_smape_df_pho1.to_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_forecsat_model/seperate_pipeline_final/xgboost_result_flake/xgboost_fs/pho_24_12_best_model.csv')

    num_lags_pho_24 = 24
    start_time = time.time()

    df_pho2, smallest_smape_df_pho2 = run_attribute_permutation_pipeline(df_pho, num_lags_pho_24, ntest_pho_24,
                                                                         feature_sel, subset_size_pho_24,
                                                                         intersect_size_pho_1, top_features_pho_25)
    df_pho2.to_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_forecsat_model/seperate_pipeline_final/xgboost_result_flake/xgboost_fs/pho_24_24_pred.csv')
    smallest_smape_df_pho2.to_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_forecsat_model/seperate_pipeline_final/xgboost_result_flake/xgboost_fs/pho_24_24_best_model.csv')

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")

    num_lags_pho_36 = 36
    start_time = time.time()

    df_pho3, smallest_smape_df_pho3 = run_attribute_permutation_pipeline(df_pho, num_lags_pho_36, ntest_pho_24,
                                                                         feature_sel, subset_size_pho_24,
                                                                         intersect_size_pho_1, top_features_pho_25)

    df_pho3.to_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_forecsat_model/seperate_pipeline_final/xgboost_result_flake/xgboost_fs/pho_24_36_pred.csv')
    smallest_smape_df_pho3.to_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_forecsat_model/seperate_pipeline_final/xgboost_result_flake/xgboost_fs/pho_24_36_best_model.csv')


    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")


    # Dallas-Fort Worth
    df_dal = pd.read_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/dal_submkt_train_test_data.csv')

    # lags = 12
    num_lags_dal_12 = 12
    ntest_dal_24 = 24
    subset_size_dal_36 = 36
    intersect_size_dal_1 = 1
    top_features_dal_37 = 37
    start_time = time.time()

    df_dal4, smallest_smape_df_dal4 = run_attribute_permutation_pipeline(df_dal, num_lags_dal_12, ntest_dal_24,
                                                                         feature_sel, subset_size_dal_36,
                                                                         intersect_size_dal_1, top_features_dal_37)
    df_dal4.to_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_forecsat_model/seperate_pipeline_final/xgboost_result_flake/xgboost_new/dal_24_12_pred.csv')
    smallest_smape_df_dal4.to_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_forecsat_model/seperate_pipeline_final/xgboost_result_flake/xgboost_new/dal_24_12_best_model.csv')

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")


    # lags = 24
    num_lags_dal_24 = 24
    ntest_dal_24 = 24
    start_time = time.time()

    df_dal5, smallest_smape_df_dal5 = run_attribute_permutation_pipeline(df_dal, num_lags_dal_24, ntest_dal_24,
                                                                         feature_sel, subset_size_dal_36,
                                                                         intersect_size_dal_1, top_features_dal_37)
    df_dal5.to_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_forecsat_model/seperate_pipeline_final/xgboost_result_flake/xgboost_new/dal_24_24_pred.csv')
    smallest_smape_df_dal5.to_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_forecsat_model/seperate_pipeline_final/xgboost_result_flake/xgboost_new/dal_24_24_best_model.csv')

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")


    # lags = 36
    num_lags_dal_36 = 36
    ntest_dal_24 = 24
    start_time = time.time()

    df_dal6, smallest_smape_df_dal6 = run_attribute_permutation_pipeline(df_dal, num_lags_dal_36, ntest_dal_24,
                                                                         feature_sel, subset_size_dal_36,
                                                                         intersect_size_dal_1, top_features_dal_37)

    df_dal6.to_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_forecsat_model/seperate_pipeline_final/xgboost_result_flake/xgboost_new/dal_24_36_pred.csv')
    smallest_smape_df_dal6.to_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_forecsat_model/seperate_pipeline_final/xgboost_result_flake/xgboost_new/dal_24_36_best_model.csv')

    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")

    df7 = pd.read_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_forecsat_model/seperate_pipeline_final/xgboost_result_flake/xgboost_fs_archive/dal_24_12_best_model.csv',
        index_col=0)
    df7['lag_num'] = 12
    df7['ntest'] = 24
    df8 = pd.read_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_forecsat_model/seperate_pipeline_final/xgboost_result_flake/xgboost_fs_archive/dal_24_24_best_model.csv',
        index_col=0)
    df8['lag_num'] = 24
    df8['ntest'] = 24
    df9 = pd.read_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_forecsat_model/seperate_pipeline_final/xgboost_result_flake/xgboost_fs_archive/dal_24_36_best_model.csv',
        index_col=0)
    df9['lag_num'] = 36
    df9['ntest'] = 24

    df7_pred = pd.read_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_forecsat_model/seperate_pipeline_final/xgboost_result_flake/xgboost_fs_archive/dal_24_12_pred.csv',
        index_col=0)
    df7_pred['lag_num'] = 12
    df7_pred['ntest'] = 24
    df8_pred = pd.read_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_forecsat_model/seperate_pipeline_final/xgboost_result_flake/xgboost_fs_archive/dal_24_24_pred.csv',
        index_col=0)
    df8_pred['lag_num'] = 24
    df8_pred['ntest'] = 24
    df9_pred = pd.read_csv(
        '/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_forecsat_model/seperate_pipeline_final/xgboost_result_flake/xgboost_fs_archive/dal_24_36_pred.csv',
        index_col=0)
    df9_pred['lag_num'] = 36
    df9_pred['ntest'] = 24

    dal_results_24 = pd.concat([df7, df8, df9]).reset_index()
    dal_smallest_smape_idx_24 = dal_results_24.groupby('research_submkt_id')['smape'].idxmin()
    dal_smallest_smape_df_24 = dal_results_24.loc[dal_smallest_smape_idx_24]
    dal_pred_results_24 = pd.concat([df7_pred, df8_pred, df9_pred]).reset_index()
    dal_best_smape_predictions_24 = dal_smallest_smape_df_24.merge(dal_pred_results_24, how='left',
                                                                   on=['research_submkt_id', 'smape', 'lag_num',
                                                                       'ntest'])


