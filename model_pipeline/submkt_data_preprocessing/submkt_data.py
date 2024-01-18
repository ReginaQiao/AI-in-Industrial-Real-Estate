import os
import warnings
import etl.connect as conn
import pandas as pd
import numpy as np
from utils.utils_preprocessing import upsample_to_monthly
from utils.utils_preprocessing import fill_missing_values_LR
from utils.fred_utils import get_national_consumption_vars
from utils.fred_utils import adjust_for_inflation


warnings.filterwarnings("ignore")


def get_popstats_data(vars=None):

    # Get popstats data from databricks
    SF = conn.SFDB(schema="BX_LINK_BXDS_GEOSPATIAL_POPSTATS")

    if vars is None:
        var_str = "*"
    else:
        var_str = ", ".join(vars)

    query_text = f"""
                select {var_str} from BX_LINK_BXDS_GEOSPATIAL_POPSTATS.POPSTATS.BLOCK_GROUP
                """

    block_pop_df = SF.read_df(query_text)
    #block_pop_df.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/block_popstats_db.csv", index=False)

    return block_pop_df


def submkt_popstats_merge(vars=None):

    # Modify popstats df to make sure merge
    if vars is None:
        vars = ['GEOID', 'DATE', 'POP_ESTIMATE_CRYR', 'HH_ESTIMATE_CRYR']

    block_pops_df = get_popstats_data(vars)

    block_pops_df['geoid'] = block_pops_df['geoid'].astype(str).apply(lambda x: int(x[1:]) if x.startswith('0') else int(x))
    # Get submarket block group level mapping data
    block_mapping_df = pd.read_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/raw data/submarket_blockgroup_mapping_with_centroids.csv", index_col=0)
    block_mapping_df = block_mapping_df[~block_mapping_df['research_submkt_id'].isin(['LIN076', 'LIN114'])]
    block_mapping_df = block_mapping_df.rename(columns={'geoid10': 'geoid'})
    block_mapping_df['geoid'] = block_mapping_df['geoid'].astype(str).apply(lambda x: int(x[1:]) if x.startswith('0') else int(x))

    # Merge two dataframes using block_group level id
    merge_df = block_mapping_df.merge(block_pops_df, on='geoid', how='left')
    #merge_df.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/block_popstats.csv")

    merge_df = merge_df.sort_values(['research_submkt_id', 'date'])
    merge_df['date'] = pd.to_datetime(merge_df['date'])
    merge_df['weighted_pop_estimate_cryr'] = merge_df['pct_area'] * merge_df['pop_estimate_cryr']
    merge_df['weighted_hh_estimate_cryr'] = merge_df['pct_area'] * merge_df['hh_estimate_cryr']
    merge_df['intptlat10'] = merge_df['pct_area'] * merge_df['intptlat10']
    merge_df['intptlon10'] = merge_df['pct_area'] * merge_df['intptlon10']
    merge_df = merge_df.groupby(['research_submkt_id','date']).agg({'weighted_pop_estimate_cryr': sum,
                                                                    'weighted_hh_estimate_cryr': sum,
                                                                    'intptlat10': np.mean,
                                                                    'intptlon10': np.mean
                                                                    }).reset_index()
    submkt_centroids_df = merge_df[['research_submkt_id','intptlon10','intptlat10']].drop_duplicates()
    merge_df = merge_df.loc[:,~merge_df.columns.isin(['intptlon10','intptlat10'])]
    merge_df = upsample_to_monthly(
        df=merge_df,
        date_col='date',
        date_index=False,
        groupby_cols=['research_submkt_id'],
        interpolate=True,
        mtd_inp=None
    )

    #merge_df.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/submkt_popstats.csv")
    #submkt_centroids_df.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/submkt_centroids.csv")

    return merge_df, submkt_centroids_df

def cbre_moodys_pops_merge():
    submkt_popstats_df, submkt_centroids_df = submkt_popstats_merge()
    submkt_popstats_df = submkt_popstats_df.reset_index()

    # moodys data - county-submarket level
    submkt_df = pd.read_csv("/mnt/container1/np_forecast_data/submarket_train_data.csv")
    # moodys data - county-market level
    #submkt_df = pd.read_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/raw data/submarket_train_data_v2.csv")

    submkt_df['date'] = pd.to_datetime(submkt_df['date'])

    submkt_df = submkt_df.merge(submkt_popstats_df, on=['date','research_submkt_id'], how='left')
    submkt_df = submkt_df.merge(submkt_centroids_df, on='research_submkt_id', how='left')

    # moodys data - county-submarket level
    #submkt_df.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/submkt_has_null_values_without_rpu.csv")
    # moodys data - county-market level
    #submkt_df.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/m_submkt_has_null_values_without_rpu.csv")

    return submkt_df


def merge_rpu_timebased_attribute():
    df = cbre_moodys_pops_merge()

    rpu = pd.read_csv('/mnt/container1/np_forecast_data/rpu_imputed.csv')

    # Select the required columns
    rpu = rpu[['research_property_id', 'research_market', 'research_submkt_id', 'total_property_sqft',
               'year_built', 'dock_doors', 'dock_door_ratio', 'number_of_car_spaces', 'parking_ratio']]

    # Group by 'research_submkt_id' and 'year_built' and aggregate the data
    grouped_df = rpu.groupby(['research_submkt_id', 'year_built']).agg({
        'dock_doors': 'sum',
        'number_of_car_spaces': 'sum',
        'total_property_sqft': 'sum'
    }).reset_index()

    # Calculate cumulative sums for 'total_dock_doors' and 'total_car_spaces'
    grouped_df['total_dock_doors'] = grouped_df.groupby('research_submkt_id')['dock_doors'].cumsum()
    grouped_df['total_car_spaces'] = grouped_df.groupby('research_submkt_id')['number_of_car_spaces'].cumsum()

    # Calculate dock_door_ratio and number_of_car_spaces_ratio
    grouped_df['dock_door_ratio'] = grouped_df['dock_doors'] / (grouped_df['total_property_sqft'] / 10000)
    grouped_df['number_of_car_spaces_ratio'] = grouped_df['number_of_car_spaces'] / (
                grouped_df['total_property_sqft'] / 1000)

    # Sort the grouped DataFrame
    grouped_df = grouped_df.sort_values(['research_submkt_id', 'year_built'])

    # Get the minimum and maximum years
    min_year = grouped_df['year_built'].min()
    max_year = grouped_df['year_built'].max()

    # Create a DataFrame with consecutive years
    consecutive_years = pd.DataFrame({'year_built': range(min_year, max_year + 1)})

    # Merge consecutive_years with unique research_submkt_id
    merged_df = pd.merge(consecutive_years, grouped_df['research_submkt_id'].drop_duplicates(), how='cross')

    # Merge grouped_df with merged_df
    merged_df = merged_df.merge(grouped_df[['research_submkt_id', 'year_built', 'total_dock_doors', 'total_car_spaces',
                                            'dock_door_ratio', 'number_of_car_spaces_ratio']], how='left',
                                on=['research_submkt_id', 'year_built'])

    # Fill null values using forward fill method
    merged_df[['total_dock_doors', 'total_car_spaces', 'dock_door_ratio', 'number_of_car_spaces_ratio']] = merged_df.groupby('research_submkt_id')[['total_dock_doors', 'total_car_spaces', 'dock_door_ratio', 'number_of_car_spaces_ratio']].ffill()
    merged_df = merged_df.rename(columns={'year_built': 'year'})
    df['year'] = pd.to_datetime(df['date']).dt.year

    merged_df = df.merge(merged_df, how='left', on=['year','research_submkt_id'])

    return merged_df


def merge_fred_attributes():
    df = merge_rpu_timebased_attribute()
    fred_df = get_national_consumption_vars()
    fred_df['a333rx1q020sbea'] = fred_df['a333rx1q020sbea'].interpolate(method='linear')
    fred_df['a333rx1q020sbea'] = fred_df['a333rx1q020sbea'].fillna(method='ffill')
    fred_df['a333rx1q020sbea'] = fred_df['a333rx1q020sbea'].fillna(method='bfill')
    ffill_cols = ['retailirsa', 'pcedg', 'pcend', 'pcepilfe', 'isratio', 'mrtsir4423xuss', 'whlslrimsa']
    fred_df[ffill_cols] = fred_df[ffill_cols].fillna(method='ffill')
    df['date'] = pd.to_datetime(df['date'])
    df = df.merge(fred_df, how='left', on='date')
    df[fred_df.columns.tolist()] = df[fred_df.columns.tolist()].fillna(method='ffill')

    # moodys data - county-submarket level
    #df.to_csv('/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/submkt_has_null_values.csv')
    # moodys data - county-market level
    #df.to_csv('/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/m_submkt_has_null_values.csv')

    return df

def get_cbre_total_property_sqft(dal):

    if dal:
        building_size = pd.read_csv(
            f'/mnt/container1/Rent_Index/ale_spreads/DallasFortWorth_cbre_submkt_id_size_category_building_class.csv')
        cbre_to_research_mapping = {
            'DAL031': 'DAL031',
            'DAL034': 'DAL034-DAL035-DAL041',
            'DAL035': 'DAL034-DAL035-DAL041',
            'DAL041': 'DAL034-DAL035-DAL041',
            'DAL037': 'DAL037-DAL038-DAL039-DAL040',
            'DAL038': 'DAL037-DAL038-DAL039-DAL040',
            'DAL039': 'DAL037-DAL038-DAL039-DAL040',
            'DAL040': 'DAL037-DAL038-DAL039-DAL040',
            'DAL042': 'DAL042-DAL043-DAL044-DAL045-DAL046',
            'DAL043': 'DAL042-DAL043-DAL044-DAL045-DAL046',
            'DAL044': 'DAL042-DAL043-DAL044-DAL045-DAL046',
            'DAL045': 'DAL042-DAL043-DAL044-DAL045-DAL046',
            'DAL046': 'DAL042-DAL043-DAL044-DAL045-DAL046',
            'DAL047': 'DAL047-DAL048',
            'DAL048': 'DAL047-DAL048',
            'DAL049': 'DAL049',
            'DAL050': 'DAL050-DAL051-DAL053-DAL054-DAL055',
            'DAL051': 'DAL050-DAL051-DAL053-DAL054-DAL055',
            'DAL053': 'DAL050-DAL051-DAL053-DAL054-DAL055',
            'DAL054': 'DAL050-DAL051-DAL053-DAL054-DAL055',
            'DAL055': 'DAL050-DAL051-DAL053-DAL054-DAL055',
            'DAL052': 'DAL052-DAL056',
            'DAL056': 'DAL052-DAL056',
            'DAL057': 'DAL057-FTW031',
            'FTW031': 'DAL057-FTW031',
            'FTW029': 'FTW029',
            'FTW032': 'FTW032-FTW033-FTW034-FTW039',
            'FTW033': 'FTW032-FTW033-FTW034-FTW039',
            'FTW034': 'FTW032-FTW033-FTW034-FTW039',
            'FTW039': 'FTW032-FTW033-FTW034-FTW039',
            'FTW035': 'FTW035-FTW036-FTW037-FTW040',
            'FTW036': 'FTW035-FTW036-FTW037-FTW040',
            'FTW037': 'FTW035-FTW036-FTW037-FTW040',
            'FTW040': 'FTW035-FTW036-FTW037-FTW040',
            'FTW038': 'FTW038-FTW041-FTW042-FTW043',
            'FTW041': 'FTW038-FTW041-FTW042-FTW043',
            'FTW042': 'FTW038-FTW041-FTW042-FTW043',
            'FTW043': 'FTW038-FTW041-FTW042-FTW043'
        }
        building_size['research_submkt_id'] = building_size['cbre_submkt_id'].map(cbre_to_research_mapping)
    else:
        building_size = pd.read_csv(
            f'/mnt/container1/Rent_Index/ale_spreads/Phoenix_cbre_submkt_id_size_category_building_class.csv')
        building_size['research_submkt_id'] = building_size['cbre_submkt_id']

    building_size['date'] = pd.to_datetime(building_size['execution_date'])
    building_size['date'] = building_size['date'] + pd.offsets.MonthBegin(1)

    grouped_df = building_size.groupby(['research_submkt_id', 'date', 'size_category']).agg({
        'n_properties': 'sum',
        'total_property_sqft': 'sum'
    }).reset_index()

    # category time series df
    category_whole_df = pd.DataFrame()
    submkt_dp = grouped_df.groupby('research_submkt_id')
    category_gp = grouped_df.groupby('size_category')
    for submkt, group_ in submkt_dp:
        for name, group in category_gp:
            cat_part = group[['research_submkt_id', 'date', 'n_properties', 'total_property_sqft']]
            cat = name
            cat_part.columns = ['research_submkt_id', 'date', f'n_properties_{cat}', f'total_property_sqft_{cat}']
            if name == grouped_df['size_category'].iloc[0]:
                category_df = cat_part.copy()
            else:
                category_df = category_df.merge(cat_part, how='left', on=['date', 'research_submkt_id'])

        category_whole_df = pd.concat([category_whole_df, category_df], axis=0)

    gp = category_whole_df.groupby('research_submkt_id')
    final_df = pd.DataFrame()
    for nm, grp in gp:
        grp = grp.drop_duplicates()
        final_df = pd.concat([final_df, grp])
    final_df = final_df.fillna(0)

    # only n_properties, sqft time series df
    sqft = grouped_df.groupby(['research_submkt_id', 'date']).agg({
        'n_properties': 'sum',
        'total_property_sqft': 'sum'
    }).reset_index()

    return final_df, sqft

def fill_null_values():
    df = merge_fred_attributes()
    df = fill_missing_values_LR(df, column='weighted_pop_estimate_cryr', grouby_col='research_submkt_id')
    df = fill_missing_values_LR(df, column='weighted_hh_estimate_cryr', grouby_col='research_submkt_id')

    return df


    cpi_adjust_cols = [
        'whlslrimsa',
        'pcedg',
        'pcend',
        'rsxfs']
    df = adjust_for_inflation(df=df, cols=cpi_adjust_cols, groupby_cols=['research_submkt_id'])
    df[cpi_adjust_cols] = df[cpi_adjust_cols].fillna(method='ffill')

    # moodys data - county-submarket level
    #df.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/submkt_data.csv")
    pho_df = df[df['research_market']=='Phoenix']
    pho_building_size_df, pho_sqft = get_cbre_total_property_sqft(False)
    pho_df = pho_df.merge(pho_building_size_df, how='left',on=['date','research_submkt_id'])
    pho_df.drop('total_property_sqft', axis=1, inplace=True)
    pho_df = pho_df.merge(pho_sqft, how='left', on=['date', 'research_submkt_id'])

    dal_df = df[df['research_market']=='Dallas-Fort Worth']
    dal_building_size_df, dal_sqft = get_cbre_total_property_sqft(True)
    dal_df = dal_df.merge(dal_building_size_df, how='left',on=['date','research_submkt_id'])
    dal_df.drop('total_property_sqft', axis=1, inplace=True)
    dal_df = dal_df.merge(dal_sqft, how='left', on=['date', 'research_submkt_id'])

    #pho_df.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/pho_submkt_data.csv")
    #dal_df.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/dal_submkt_data.csv")

    pho_df_train = pho_df[pho_df['split'] == 'train']
    pho_df_pred = pho_df[pho_df['split'] == 'test']
    pho_df_train.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/pho_submkt_train_test_data.csv")
    pho_df_pred.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/pho_submkt_pred_data.csv")

    dal_df_train = dal_df[dal_df['date'] <= '2023-07-01']
    dal_df_pred = dal_df[dal_df['date'] > '2023-07-01']
    dal_df_train.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/dal_submkt_train_test_data.csv")
    dal_df_pred.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/dal_submkt_pred_data.csv")


    # moodys data - county-market level
    #df.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/m_submkt_data.csv")
    #pho_df = df[df['research_market'] == 'Phoenix']
    #dal_df = df[df['research_market'] == 'Dallas-Fort Worth']
    #pho_df.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/m_pho_submkt_data.csv")
    #dal_df.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/m_dal_submkt_data.csv")

    #pho_df_train = pho_df[pho_df['split'] == 'train']
    #pho_df_pred = pho_df[pho_df['split'] == 'test']
    #pho_df_train.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/m_pho_submkt_train_test_data.csv")
    #pho_df_pred.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/m_pho_submkt_pred_data.csv")

    #dal_df_train = dal_df[dal_df['split'] == 'train']
    #dal_df_pred = dal_df[dal_df['split'] == 'test']
    #dal_df_train.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/m_dal_submkt_train_test_data.csv")
    #dal_df_pred.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/submkt_train_data/m_dal_submkt_pred_data.csv")

    return dal_df_train


if __name__ == "__main__":
    df = fill_null_values()

    print(df.shape)
