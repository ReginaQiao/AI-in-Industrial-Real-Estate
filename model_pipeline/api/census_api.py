import pandas as pd
import numpy as np
import requests

"""
Block Group: STATE+COUNTY+TRACT+BLOCK GROUP
2+3+6+1=12
"""
county_submkt_mapping = pd.read_csv('/mnt/container1/np_forecast_data/submarket_county_mapping.csv')
county_sub_submkt_mapping = pd.read_csv('/mnt/container1/np_forecast_data/submarket_county_sub_mapping.csv')
county_submkt_mapping[county_submkt_mapping['research_market']=='Phoenix']['research_submkt_id'].unique()
county_submkt_mapping[county_submkt_mapping['research_market']=='Dallas-Fort Worth']['research_submkt_id'].unique()

pho_counties = county_submkt_mapping[(county_submkt_mapping['research_submkt_id']!='LIN114')&(county_submkt_mapping['research_market']=='Phoenix')]['county_geoid'].unique().tolist()
dal_counties = county_submkt_mapping[(county_submkt_mapping['research_submkt_id']!='LIN076')&(county_submkt_mapping['research_market']=='Dallas-Fort Worth')]['county_geoid'].unique().tolist()

pho_countysub_fips = county_sub_submkt_mapping[(county_sub_submkt_mapping['research_submkt_id']!='LIN114')&(county_sub_submkt_mapping['research_market']=='Phoenix')]['cousubfp'].unique().tolist()
dal_countysub_fips = county_sub_submkt_mapping[(county_sub_submkt_mapping['research_submkt_id']!='LIN076')&(county_sub_submkt_mapping['research_market']=='Dallas-Fort Worth')]['cousubfp'].unique().tolist()

pho_countysub_fips = [str(fip) for fip in pho_countysub_fips]
dal_countysub_fips = [str(fip) for fip in dal_countysub_fips]

pho_countysub_fips_str = ','.join(pho_countysub_fips)
dal_countysub_fips_str = ','.join(dal_countysub_fips)

def get_county_fips_code(state_fips_code):
    counties_fips_code = []

    if state_fips_code == 4:
        for num in pho_counties:
            if str(num).startswith('4'):
                counties_fips_code.append(str(num)[1:])
    elif state_fips_code == 48:
        for num in dal_counties:
            if str(num).startswith('48'):
                counties_fips_code.append(str(num)[2:])
    return counties_fips_code


def get_countysub_data(
        year: str = None,
        variables: str = None,
        state_fips: str = None,
        countysub_fips: list = None):
    base_url = 'https://api.census.gov/data'
    dataset = 'acs/acs1'

    if year is None:
        print('Please input variable: year.')
    year = year

    if variables is None:
        print('Please input variable: variables.')
    variables = variables

    level = 'county%20subdivision'

    if state_fips is None:
        print('Please input varaible: state_fips.')

    state_fips = state_fips

    if countysub_fips is None:
        print('Please input varaible: countysub_fips.')

    countysub_fips = countysub_fips

    api_key = '2caac992889c23e9fe800fc4a11ffd2248146d2c'

    url = f'{base_url}/{year}/{dataset}/profile?get={variables}&for={level}:*&in=state:{state_fips}&in=county:*&key={api_key}'

    r = requests.get(url)

    try:
        resp = r.json()
        if r.status_code == 200:
            data = resp[1:]
            df = pd.DataFrame(data, columns=resp[0])
            df['countysub_geoid'] = df['state'] + df['county'] + df['county subdivision']
            df['countysub_geoid'] = df['countysub_geoid'].astype(str).apply(
                lambda x: int(x[1:]) if x.startswith('0') else int(x))
            dt = f'{year}-01-01'
            df['date'] = pd.Series([dt] * df.shape[0])
            df['date'] = pd.to_datetime(df['date'])
            df = df[df['county subdivision'].isin(countysub_fips)]
            df = df.drop(['state', 'county', 'county subdivision'], axis=1)

            return df
        else:
            print("Error occurred while retrieving data.")
    except ValueError:
        print(f"{year}:Response content is not valid JSON.")


def get_blockgroup_data(
        year: str = None,
        variables: str = None,
        state_fips: str = None,
        county_fips: str = None):
    base_url = 'https://api.census.gov/data'
    dataset = 'acs/acs5'

    if year is None:
        print('Please input variable: year.')
    year = year

    if variables is None:
        print('Please input variable: variables.')
    variables = variables

    level = 'block%20group'

    if state_fips is None:
        print('Please input varaible: state_fips.')

    state_fips = state_fips + '%20'

    if county_fips is None:
        print('Please input varaible: county_fips.')

    county_fips = county_fips

    api_key = '2caac992889c23e9fe800fc4a11ffd2248146d2c'

    url = f'{base_url}/{year}/{dataset}?get={variables}&for={level}:*&in=state:{state_fips}&in=county:{county_fips}&key={api_key}'

    r = requests.get(url)

    try:
        resp = r.json()
        if r.status_code == 200:

            data = resp[1:]
            df = pd.DataFrame(data, columns=resp[0])
            df['blockgroup_geoid'] = df['state'] + df['county'] + df['tract'] + df['block group']
            df['blockgroup_geoid'] = df['blockgroup_geoid'].astype(str).apply(
                lambda x: int(x[1:]) if x.startswith('0') else int(x))
            dt = f'{year}-01-01'
            df['date'] = pd.Series([dt] * df.shape[0])
            df['date'] = pd.to_datetime(df['date'])
            df = df.drop(['state', 'county', 'tract', 'block group'], axis=1)

            return df
        else:
            print("Error occurred while retrieving data.")
    except ValueError:
        print(f"{year}:Response content is not valid JSON.")


def get_asc_blockgroup_data(year, variables, state_fips, county_fips):
    dfs = pd.DataFrame()

    for yr in year:
        df = get_blockgroup_data(yr, variables, state_fips, county_fips)
        dfs = pd.concat([dfs, df])

    return dfs


if __name__ == "__main__":

    pho_fips_codes = get_county_fips_code(4)
    dal_fips_codes = get_county_fips_code(48)

    pho_fips_codes_str = ','.join(pho_fips_codes)
    dal_fips_codes_str = ','.join(dal_fips_codes)

    year = [str(yr) for yr in range(2014, 2022)]
    variables = 'B01001_001E'

    # get Phoenix county_subdivision population data
    state_fips = '04'
    county_fips = pho_fips_codes_str
    pho_bg_df = get_asc_blockgroup_data(year, variables, state_fips, county_fips)
    pho_bg_df.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/pho_csa_blockgroup.csv")

    # get Phoenix county_subdivision population data
    state_fips = '48'
    county_fips = dal_fips_codes_str

    dal_bg_df = get_asc_blockgroup_data(year, variables, state_fips, county_fips)
    dal_bg_df.to_csv("/mnt/container1/zqiao_Workspace/link-research/ad-hoc/zq-sandbox/submkt_data/dal_csa_blockgroup.csv")
















