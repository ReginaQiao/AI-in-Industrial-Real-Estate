import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import multiprocessing as mp
from typing import List, Optional
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from xgboost import XGBRegressor


def all_split_data_by_submarket(data, ntest):
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
    df_new = df[['date', 'research_submkt_id', 'real_hedonic_rent_submarket']]
    df_new['date'] = pd.to_datetime(df_new['date'])
    df['date'] = pd.to_datetime(df['date'])

    if num_lags is None:
        num_lags = 36

    for lag in range(1, num_lags + 1):
        df_new['rent_{}months_ago'.format(lag)] = df_new.groupby('research_submkt_id')[
            'real_hedonic_rent_submarket'].shift(lag)
    df_new = df_new.dropna()
    df_new = df_new.sort_values(['date', 'research_submkt_id']).reset_index(drop=True)
    df_sel = df[feature_subset + ['date', 'research_submkt_id']]
    df_new = df_new.merge(df_sel, on=['date', 'research_submkt_id'], how='left')

    X = df_new.iloc[:, [0, 1] + list(range(3, len(df_new.columns)))]
    Y = df_new.iloc[:, :3]

    Y_train, Y_test = all_split_data_by_submarket(Y, ntest)
    y_train = Y_train.iloc[:, -1]
    y_test = Y_test.iloc[:, -1]
    X_train, X_test = all_split_data_by_submarket(X, ntest)
    x_train = X_train.iloc[:, 2:]
    x_test = X_test.iloc[:, 2:]

    param_grid = {
        'n_estimators': [100, 150],  # ,200, 250, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.15, 0.1, 0.01, 0.001],
        # 'gamma': [0, 0.1, 0.2, 0.3],
        # 'colsample_bytree': [0.5, 0.75, 1],
        # 'reg_alpha': [0, 0.1, 0.5, 1],
        # 'reg_lambda': [0, 0.1, 0.5, 1]
    }

    model = XGBRegressor()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring=make_scorer(smape))
    grid_search.fit(x_train, y_train)

    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_
    # print("Best Parameters:", best_params)
    # print("Best Score (SMAPE):", best_score)

    best_model = XGBRegressor(**best_params)
    best_model.fit(x_train, y_train)

    y_pred = best_model.predict(x_test)

    Y_test_pred = Y_test.copy()
    Y_test_pred['y_pred'] = y_pred

    whole_smape = smape(y_test, y_pred)

    smape_dic = {}
    submkt_id = df_new['research_submkt_id'].unique().tolist()
    for submkt in submkt_id:
        y_submkt_test = Y_test_pred[Y_test_pred['research_submkt_id'] == submkt]['real_hedonic_rent_submarket']
        y_submkt_pred = Y_test_pred[Y_test_pred['research_submkt_id'] == submkt]['y_pred']
        submkt_smape = smape(y_submkt_test, y_submkt_pred)
        smape_dic[submkt] = submkt_smape

    # print(f"{submkt} - SMAPE: {smape_value:.2f}%")

    return Y_test_pred, smape_dic, whole_smape


def get_feature_subsets(
        feature_space,
        subset_size=2,
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


def run_attribute_permutation_pipeline(df, num_lags, ntest, feature_space, subset_size, intersect_size):

    feature_subsets = get_feature_subsets(
        feature_space,
        subset_size=subset_size,
        include_features=None,
        intersect_size=intersect_size
    )

    pool = mp.Pool(processes=mp.cpu_count())
    results = []

    for features in feature_subsets:
        result = pool.apply_async(all_get_submkt_forecast, kwds={
            "df": df,
            "num_lags": num_lags,
            "ntest": ntest,
            "feature_subset": features,
            # Pass other required parameters here
        })
        results.append(result)

    pool.close()
    pool.join()

    #smape_df_list = []
    #for result in results:
    #    Y_test_pred, smape_dic = result.get()
    #    smape_df = pd.DataFrame.from_dict(smape_dic, orient='index', columns=['smape'])
    #    smape_df['research_submkt_id'] = smape_df.index
    #    smape_df_list.append(smape_df)

    #smape_df = pd.concat(smape_df_list, ignore_index=True)

    #best_subset = None
    #best_score = float('inf')
    #for result in results:
    #    subset_score = result.get()
    #    if subset_score < best_score:
    #        best_subset = subset
    #        best_score = subset_score

    return results #smape_df #best_subset


def all_plot_submkt_forecast(Y_test_pred, submkt_id):
    x = Y_test_pred[Y_test_pred['research_submkt_id'] == submkt_id]['date']
    y = Y_test_pred[Y_test_pred['research_submkt_id'] == submkt_id]['real_hedonic_rent_submarket']
    y_pred = Y_test_pred[Y_test_pred['research_submkt_id'] == submkt_id]['y_pred']

    plt.plot(x, y, label='test')
    plt.plot(x, y_pred, label='pred')
    plt.title('{} submkt_rent forecasting'.format(submkt_id))
    plt.legend()

    return plt.show()


def get_single_smape(Y_test_pred, submkt_id):
    y = Y_test_pred[Y_test_pred['research_submkt_id'] == submkt_id]['real_hedonic_rent_submarket']
    y_pred = Y_test_pred[Y_test_pred['research_submkt_id'] == submkt_id]['y_pred']

    smp = smape(y, y_pred)

    return smp


def get_whole_smape_df(Y_test_pred, submkt_list):
    smp_ls = []
    for submkt_id in submkt_list:
        y = Y_test_pred[Y_test_pred['research_submkt_id'] == submkt_id]['real_hedonic_rent_submarket']
        y_pred = Y_test_pred[Y_test_pred['research_submkt_id'] == submkt_id]['y_pred']
        smp = smape(y, y_pred)
        smp_ls.append(smp)
    smp_dic = {'research_submkt_id': submkt_list,
               'smape': smp_ls}
    smp_df = pd.DataFrame(smp_dic)

    return smp_df


if __name__ == "__main__":
    data = pd.read_csv('/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/pho_submkt_train_test_data.csv')
    feature_space = [
        "real_market_level_rent",
        "gdp_histfc",
        "nominal_retail_sales_histfc",
        #"employment_histfc",
        #"real_ecommerce",
        #"spread_3m10y",
        #"real_retail_sales_ex_gas",
        #"imports_us",
        #"ecomm^2_pop",
        #"weighted_pop_estimate_cryr",
        #"weighted_hh_estimate_cryr"
    ]
    subset_size = 2
    intersect_size = 1

    # Run the attribute permutation pipeline
    ntest = 24
    best_subset = run_attribute_permutation_pipeline(data, ntest, feature_space, subset_size, intersect_size)

    print("Best Subset:", best_subset)

