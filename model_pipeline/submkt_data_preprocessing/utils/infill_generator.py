#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 13:03:34 2021

@author: mrand
"""



## set up the environment
import pandas as pd 
import geopandas as gpd
import numpy as np 
# import matplotlib.pyplot as plt
import contextily as ctx
from scipy import stats
#import libpysal
#import tobler
import fiona
from scipy.sparse import data 
#from tobler.util import h3fy
from tobler.area_weighted import area_interpolate
#from cenpy import products
#import seaborn as sns 
#sns.set_theme(style="ticks")
# import geoplot
# import mapclassify
# from haversine import haversine, Unit


mi_to_km = 0.621371
acres_to_sqmi = 640
sqft_per_acre = 43560


blob_folder = '/mnt/container1/'





### Bring in the cbsa mapping file
cbsa2fipsxw = pd.read_csv(blob_folder +'cbsa2fipsxw.csv', header=0, skip_blank_lines=True) 
##skip_blank_lines=True

cbsa2fipsxw['state_county_fips'] = cbsa2fipsxw['fipsstatecode'].astype(str).str.zfill(2)+cbsa2fipsxw['fipscountycode'].astype(str).str.zfill(3)

cbsa2fipsxw['csa_cbsa_code'] = np.where(cbsa2fipsxw['csacode'].isnull(), cbsa2fipsxw['cbsacode'],cbsa2fipsxw['csacode'])
cbsa2fipsxw['csa_cbsa_title'] = np.where(cbsa2fipsxw['csacode'].isnull(), cbsa2fipsxw['cbsatitle'],cbsa2fipsxw['csatitle'])

unique_csacbsa = sorted(cbsa2fipsxw[['csa_cbsa_title']].drop_duplicates().values)
#display(cbsa2fipsxw)

cbsa2fipsxw['cbsa_metdiv_code'] = np.where(cbsa2fipsxw['metrodivisioncode'].isnull(), cbsa2fipsxw['cbsacode'],cbsa2fipsxw['metrodivisioncode'])
cbsa2fipsxw['cbsa_metdiv_title'] = np.where(cbsa2fipsxw['metrodivisioncode'].isnull(), cbsa2fipsxw['cbsatitle'],cbsa2fipsxw['metropolitandivisiontitle'])


# import zipfile
# with zipfile.ZipFile(blob_folder+'Major_Ports.zip', 'r') as zip_ref:
#     zip_ref.extractall(blob_folder)

### Read the shapefile

shp = gpd.read_file(
    blob_folder +'nhgis0048_shape/nhgis0048_shapefile_tl2019_us_blck_grp_2019/US_blck_grp_2019.shp'
)
shp.columns = shp.columns.str.lower()
shp['state_county_fips'] = shp['statefp'].astype(str).str.zfill(2)+shp['countyfp'].astype(str).str.zfill(3)
shp = shp.merge(cbsa2fipsxw, on="state_county_fips", how='left')

### limit the shp to the subject metro
# subject_csacbsa = ['Los Angeles-Long Beach, CA'] ## can select more than one here
# shp = shp[shp['csa_cbsa_title'].isin(subject_csacbsa)]

### convert to 4326
shp = shp.to_crs("EPSG:4326")


 
    
##################################

statefp_lst = shp['statefp'].astype(int).drop_duplicates().values
countyfp_lst = shp['countyfp'].astype(int).drop_duplicates().values
gisjoin_lst = shp['gisjoin'].drop_duplicates().values


acs_filename = blob_folder + 'nhgis0049_ds244_20195_2019_blck_grp.csv'

## load in chunks, making sure only the metro in question shows up here
iter_csv = pd.read_csv(acs_filename, header = 0, encoding = "ISO-8859-1", iterator=True, chunksize=1000, #dtype='object',
                      low_memory=False)
acs_df = pd.concat([chunk[(chunk['STATEA'].isin(statefp_lst)) &
                      (chunk['COUNTYA'].isin(countyfp_lst)) ] for chunk in iter_csv])
    
#acs_df = pd.read_csv(acs_filename, header = 0, encoding = "ISO-8859-1")


#acs_df = acs_df[(acs_df['STATEA'].isin(statefp_lst)) &
 #                    (acs_df['COUNTYA'].isin(countyfp_lst)) ]
    
acs_df.columns = acs_df.columns.str.lower()
acs_df = acs_df[acs_df['gisjoin'].isin(gisjoin_lst)]

tot_pop = 'alube001' #'B01003'
hh = 'alu9e001'
med_hh_inc = 'alw1e001' #ALW1M001
hu = 'alzje001'
owner_hu = 'alzle002'
tot_occ_hu = 'alzle001' ## denominator for homeownership
neighborhood_median_yr_built = 'al0ee001'
median_owned_value = 'al1he001'
edu_bachelors = 'alwge022'
edu_masters = 'alwge023'
edu_professional = 'alwge024'
edu_doctorate = 'alwge025'
edu_denom = 'alwge001'
poverty_hh = 'alwye002'
poverty_hh_denom = 'alwye001' 
labor_force = 'aly3e002'


acs_df = acs_df[['gisjoin', tot_pop, hh, med_hh_inc, hu, owner_hu,
                 tot_occ_hu, neighborhood_median_yr_built, median_owned_value, edu_bachelors,
                 edu_masters, edu_professional, edu_doctorate, edu_denom, poverty_hh, poverty_hh_denom,
                 labor_force]]
acs_df.columns = ['gisjoin', 'tot_pop', 'hh', 'med_hh_inc', 'hu',
                  'owner_hu', 'tot_occ_hu', 'neighborhood_median_yr_built', 'median_owned_value', 'edu_bachelors',
                 'edu_masters', 'edu_professional', 'edu_doctorate', 'edu_denom', 'poverty_hh', 'poverty_hh_denom',
                 'labor_force']


acs_df['college_plus'] = acs_df['edu_bachelors'] + acs_df['edu_masters'] + acs_df['edu_professional'] + acs_df['edu_doctorate']
acs_df['masters_plus'] = acs_df['edu_masters'] + acs_df['edu_professional'] + acs_df['edu_doctorate']
acs_df = acs_df.drop(columns = ['edu_bachelors', 'edu_masters', 'edu_professional', 'edu_doctorate'])
acs_df['med_hh_inc_wt'] = acs_df['med_hh_inc'] * acs_df['hh']
acs_df['median_owned_value_wt'] = acs_df['median_owned_value'] * acs_df['owner_hu']
acs_df['neighborhood_median_yr_built_denom'] = np.where((acs_df.neighborhood_median_yr_built == 0),0,acs_df.hu)
acs_df['neighborhood_median_yr_built_wt'] = acs_df['neighborhood_median_yr_built'] * acs_df['neighborhood_median_yr_built_denom']

shp = shp.merge(acs_df, on="gisjoin")
del acs_df ## free up memory
acs_agg = shp[['csa_cbsa_title', 'aland', 'awater', 'tot_pop', 'hh', 'hu', 'med_hh_inc_wt',
               'labor_force']].groupby('csa_cbsa_title').sum()
acs_cbsa_agg = shp[['cbsacode', 'aland', 'awater', 'tot_pop', 'hh', 'hu', 'med_hh_inc_wt',
                    'labor_force']].groupby('cbsacode').sum()
acs_cbsa_agg['cbsa_metdiv_code'] = acs_cbsa_agg.index
acs_cbsa_agg.index = range(0, len(acs_cbsa_agg))

acs_metdiv_agg = shp[['metrodivisioncode', 'aland', 'awater', 'tot_pop', 'hh', 'hu', 'med_hh_inc_wt',
                      'labor_force']].groupby('metrodivisioncode').sum()
acs_metdiv_agg['cbsa_metdiv_code'] = acs_metdiv_agg.index
acs_metdiv_agg.index = range(0, len(acs_metdiv_agg))

acs_cbsa_metdiv_agg = pd.concat([acs_cbsa_agg, acs_metdiv_agg])




#################################################################




epa_filename = blob_folder +'SmartLocationDatabase.gdb' ##'SmartLocationDb/SMARTLOCATIONDB/SmartLocationDb.gdb'
## this has d5ae but not % of pop



# Get all the layers from the .gdb file 
layers = fiona.listlayers(epa_filename)

for layer in layers:
    epa = gpd.read_file(epa_filename,layer=layer)
    # Do stuff with the gdf

epa.columns = epa.columns.str.lower()
epa = epa.replace(-99999, np.nan) ## wipe out the -99999s
epa = epa.to_crs(shp.crs)

epa['state_county_fips'] = epa['statefp'].astype(str).str.zfill(2)+epa['countyfp'].astype(str).str.zfill(3)
epa = epa.merge(cbsa2fipsxw, on="state_county_fips")

#drive_var = "d5ae"
### extensive = sum cols
### intensive = wt cols

epa_extensive = ['ac_tot','ac_unpr','ac_water','ac_land','counthu','totpop','autoown0','autoown1','autoown2p',
                'workers','totemp','e8_ret','e_lowwagewk','e_medwagewk','e_hiwagewk'                 
                ]

epa_intensive = ['p_wrkage', 'pct_ao0', 'pct_ao1', 'pct_ao2p', 'd1a', 'd1b', 'd1c', 'd1c5_ind','d1c5_ret', 'd1d',
                'd2a_jphh', 'd2c_trpmx1', 'd3a', 'd3aao',
                'd3amm', 'd3apo', 'd3b', 'd3bao', 'd3bmm3', 'd3bmm4', 'd3bpo3',
                 'd3bpo4', 'd4a', 'd4b025', 'd4b050', 'd4c', 'd4d', 'd5cr', 'd5cri',
                 'd5ce', 'd5cei', 'd5dr', 'd5dri', 'd5de', 'd5dei',
                 'd5ar', 'd5ae', 'd5br' , 'd5be',
                 'vmt_per_worker','com_vmt_per_worker','gasprice',
                 'natwalkind'
                ]

# categorical_variables
epa_categorical = []


acs_extensive = ['tot_pop', 'hh', 'hu', 'med_hh_inc_wt', 'aland', 'awater',
                  'median_owned_value_wt','neighborhood_median_yr_built_wt','neighborhood_median_yr_built_denom',
                  'owner_hu', 'tot_occ_hu',  'college_plus', 'masters_plus',
                  'edu_denom', 'poverty_hh', 'poverty_hh_denom']
acs_intensive = ['med_hh_inc', 'median_owned_value','neighborhood_median_yr_built']
acs_categorical = ['csa_cbsa_title','csa_cbsa_code','cbsacode','cbsatitle','statefp','countyfp']

shp[acs_extensive] = shp[acs_extensive].replace(np.nan, 0) ## wipe out the nans



### pulling in the same flat CRS from the Tobler tutorial
## i know, i know, it's a lazy CRS pickup
precincts = gpd.read_file("https://ndownloader.figshare.com/files/20460549")
#precincts.crs


### create population density at the CSA_CBSA level

epa_cbsa_agg = epa[['cbsacode', 'ac_unpr']].groupby('cbsacode').sum()
epa_cbsa_agg['cbsa_metdiv_code'] = epa_cbsa_agg.index
epa_cbsa_agg.index = range(0, len(epa_cbsa_agg))

epa_metdiv_agg = epa[['metrodivisioncode', 'ac_unpr']].groupby('metrodivisioncode').sum()
epa_metdiv_agg['cbsa_metdiv_code'] = epa_metdiv_agg.index
epa_metdiv_agg.index = range(0, len(epa_metdiv_agg))

epa_cbsa_metdiv_agg = pd.concat([epa_cbsa_agg, epa_metdiv_agg])
acs_cbsa_metdiv_agg = acs_cbsa_metdiv_agg.merge(epa_cbsa_metdiv_agg, how='left', on='cbsa_metdiv_code')
acs_cbsa_metdiv_agg['pop_density'] = acs_cbsa_metdiv_agg['tot_pop'] / acs_cbsa_metdiv_agg['ac_unpr']

# epa['state_county_fips'] = epa['SFIPS'].astype(str).str.zfill(2)+epa['CFIPS'].astype(str).str.zfill(3)
# epa = epa.merge(cbsa2fipsxw, on="state_county_fips")
epa_agg = epa[['csa_cbsa_title', 'ac_unpr']].groupby('csa_cbsa_title').sum()
acs_agg = acs_agg.merge(epa_agg, how='left', on='csa_cbsa_title')

# acs_agg = acs_agg[acs_agg['ac_unpr'] > 0] ## pulls out AK/HI
acs_agg['pop_density'] = acs_agg['tot_pop'] / acs_agg['ac_unpr']
acs_agg['csa_cbsa_title'] = acs_agg.index
acs_agg.index = range(0, len(acs_agg))

# acs_agg['tot_pop_q'] = pd.qcut(acs_agg['tot_pop'], q=10, labels=range(1, 11))
# acs_agg['pop_density_q'] = pd.qcut(acs_agg['pop_density'], q=10, labels=range(1, 11))

###################################################################################

##### this is the center points file with drivetimes from BX/HERE
### center points
centerpts = pd.read_csv(blob_folder+'centerpts-drivetimes.csv', header=0, skip_blank_lines=True) 
centerpts.columns = centerpts.columns.str.lower()
centerpts['cbsa_metdiv_code'] = np.where(centerpts['metrodivisioncode'].isnull(), centerpts['cbsacode'],centerpts['metrodivisioncode'])
centerpts['cbsa_metdiv_title'] = np.where(centerpts['metrodivisioncode'].isnull(), centerpts['cbsatitle'],centerpts['metropolitandivisiontitle'])

### let's start measuring how important / big these are
centerpts = centerpts.merge(acs_cbsa_metdiv_agg[['cbsa_metdiv_code', 'tot_pop', 'pop_density']], how='left', on='cbsa_metdiv_code')
centerpts = centerpts.rename(columns={'tot_pop': 'tot_pop_cbsa_metdiv', 'pop_density': 'pop_density_cbs_metdiv'})
 
# ## warning! these are non-unique
centerpts['csa_cbsa_title'] = np.where(centerpts['csacode'].isnull(), centerpts['cbsatitle'],centerpts['csatitle'])

# ### this brings in population density (per acre) at the csa_cbsa level
centerpts = centerpts.merge(acs_agg[['csa_cbsa_title', 'tot_pop', 'pop_density']], how='left', on='csa_cbsa_title')
centerpts['pop_to_tot_pop'] = centerpts['tot_pop_cbsa_metdiv'] / centerpts['tot_pop']
centerpts['pop_density_to_tot'] = centerpts['pop_density_cbs_metdiv'] / centerpts['pop_density']



# centerpts = centerpts.sort_values(by='pop_density_cbs_metdiv', ascending=False)
# centerpts['pop_density_q'] = pd.qcut(-centerpts['pop_density_cbs_metdiv'], q=10, labels=range(1, 11))
# centerpts['tot_pop_q'] = pd.qcut(-centerpts['tot_pop_cbsa_metdiv'], q=10, labels=range(1, 11))

 
# centerpts['q_mult'] = centerpts['pop_density_q'] * centerpts['tot_pop_q'] 
centerpts['pop_mult_tot'] = centerpts['pop_density'] * centerpts['tot_pop'] 
centerpts['pop_mult_cbsa_metdiv'] = centerpts['pop_density_cbs_metdiv'] * centerpts['tot_pop_cbsa_metdiv'] 

centerpts['pop_mult'] = centerpts['tot_pop_cbsa_metdiv'] #* (centerpts['pop_density_cbs_metdiv']) ##**centerpts['pop_density_to_tot']
centerpts = centerpts.sort_values(by='pop_mult', ascending=False)
centerpts['pop_mult_q'] = pd.qcut(-centerpts['pop_mult'], q=10, labels=range(1, 11))

# temp_q = pd.DataFrame(centerpts[['csa_cbsa_title','pop_density']].drop_duplicates())
# temp_q['pop_density_q'] = pd.qcut(-temp_q['pop_density'], q=10, labels=range(1, 11))
# centerpts = centerpts.merge(temp_q[['csa_cbsa_title', 'pop_density_q']], how='left', on='csa_cbsa_title')

## choose which density quantile gets which drive-time
dens_q_df = pd.DataFrame({
    'pop_mult_q' : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'drive_time' : [60, 45, 45, 45, 30, 30, 30, 30, 15, 15]
    })
dens_q_df['polygon_text'] = str('here_truck_disabled_')
dens_q_df['polygon_mult'] = dens_q_df['drive_time'] * 60


centerpts = centerpts.merge(dens_q_df[['pop_mult_q', 'drive_time']], how='left', on='pop_mult_q')
centerpts.to_csv(blob_folder + 'PythonOut_infill_centerpts_drivetimes.csv')



drive_list = dens_q_df['drive_time'].drop_duplicates().to_list()
drivetimes = []

for d in drive_list:
    c1 = centerpts[centerpts['drive_time'] == d] 
    c1 = c1[['cbsa_metdiv_title', str('here_truck_disabled_'+ str(60*d))]]
    c1.columns = ['cbsa_metdiv_title', 'drive_polygon']
    drivetimes.append(c1)

drivetimes = pd.concat(drivetimes)
centerpts = centerpts.merge(drivetimes, how='left', on='cbsa_metdiv_title')


centerpts_gdf = gpd.GeoDataFrame(centerpts, geometry=gpd.points_from_xy(centerpts.lon_cbd, centerpts.lat_cbd))
centerpts_gdf.crs = shp.crs




centershapes = pd.DataFrame(centerpts.drop(columns='geometry'))
centershapes['geometry'] = gpd.GeoSeries.from_wkt(centershapes['drive_polygon'])
drivetime_poly = gpd.GeoDataFrame(centershapes)
drivetime_poly.crs = shp.crs





### Airports
### 5/24/21: web site is down. reverting to saved version -- resaved 4/1/2022
# airports = pd.read_csv('https://ourairports.com/data/airports.csv',
#                        header=0, skip_blank_lines=True)

airports = pd.read_csv(blob_folder+'airports.csv',
                       header=0, skip_blank_lines=True)
# airports = ourairports().

airports = gpd.GeoDataFrame(airports, geometry=gpd.points_from_xy(airports.longitude_deg, airports.latitude_deg))
airports.crs = shp.crs



air_freight = pd.read_csv(blob_folder+'Airport_Freight.csv',
                       header=0, skip_blank_lines=True)


air_intermodal = pd.read_csv(blob_folder+'air_freight_intermodal.csv',
                       header=0, skip_blank_lines=True)

air_includes = air_freight['Airport_Code'].drop_duplicates().to_list()
air_includes1 = air_intermodal['LOCID'].drop_duplicates().to_list()
air_includes = list(set(air_includes + air_includes1))


airports = airports[airports['iata_code'].isin(air_includes)]
airports = airports.merge(air_freight, how='left', left_on='iata_code', right_on='Airport_Code')
airports = airports.merge(air_intermodal, how='left', left_on='iata_code', right_on='LOCID')

airports['EST_AREA'] = pd.to_numeric(airports['EST_AREA'], errors = 'coerce')

#################################################################
ports_filename = blob_folder +'Major_Ports'

# Get all the layers from the .gdb file 
layers = fiona.listlayers(ports_filename)

for layer in layers:
    ports = gpd.read_file(ports_filename,layer=layer)
    # Do stuff with the gdf
    
if len(ports[ports.is_valid == False]) > 0:
    mask = ports.is_valid == False#['geometry']
    s_invalid = ports[mask]
    newval = gpd.GeoDataFrame(s_invalid, geometry=s_invalid.buffer(0))
    ports.loc[mask, 'geometry'] = newval['geometry']
    del(mask)
    del(newval)
#################################################################



#################################################################
rail_intermodal_filename = blob_folder +'Intermodal_Freight_Facilities_RailTOFCCOFC'

# Get all the layers from the .gdb file 
layers = fiona.listlayers(rail_intermodal_filename)

for layer in layers:
    rail_intermodal = gpd.read_file(rail_intermodal_filename,layer=layer)
    # Do stuff with the gdf
    
if len(rail_intermodal[rail_intermodal.is_valid == False]) > 0:
    mask = rail_intermodal.is_valid == False#['geometry']
    s_invalid = rail_intermodal[mask]
    newval = gpd.GeoDataFrame(s_invalid, geometry=s_invalid.buffer(0))
    rail_intermodal.loc[mask, 'geometry'] = newval['geometry']
    del(mask)
    del(newval)
#################################################################




### Amazon facilities
amazon =  pd.read_excel(
    blob_folder+'Amazon Distribution Network V10.0 2021-11-01 - Short.xlsx',
    sheet_name='1',
    skiprows=4,
    engine = 'openpyxl'
)

fix_cols = [str(i).replace('\n', ' ') for i in amazon.columns]
amazon.columns = fix_cols
amazon = amazon.dropna(subset=['Unique Record Serial Number'])
amazon = amazon[amazon['Country']=='USA']
amazon = amazon[~amazon['Facility Open (Y/N)'].isin(['C'])]
# amazon = amazon[~amazon['Facility Existing or New'].isin(['Cancelled', 'Closed'])]


# amazon_obj = amazon.select_dtypes(['object'])
# amazon[amazon_obj.columns] = amazon_obj.apply(lambda x: x.str.strip())
amazon_includes = ['Sortable', 'Non-Sortable', 'Returns',
                   'Prime', 'Inbound Cross Dock', 'Air hub',
                   'Sortation Center', 'Delivery Station']
amazon = amazon[amazon['Facility Type'].isin(amazon_includes)]

amazon['Total Working Square  Feet'] = amazon['Total Working Square  Feet'].replace(np.nan, 0)
amazon['Annual Packages Shipped'] = amazon['Annual Packages Shipped'].replace(np.nan, 0)

amazon = gpd.GeoDataFrame(amazon, geometry=gpd.points_from_xy(amazon.Longitude, amazon.Latitude))
amazon.crs = shp.crs




#Pulling in FedEx data
fedex = pd.read_excel(
    blob_folder+'MWPVL International Report on FedEx North America Operations 2022.xlsb',
    sheet_name='1',
    engine='pyxlsb',
    header=3
    )[15:]

fix_cols = [str(i).replace('\n', ' ') for i in fedex.columns]
fedex.columns = fix_cols

fedex = fedex[fedex['Country.1']=='USA'] #filter out non-American facilities
# fedex = fedex[fedex['Active']=='Y'] #filter out inactive facilities
# fedex = fedex[~fedex['Facility Type (Primary Role)'].isin(['FedEx Ship Center'])] #filter out retail stores

fedex = gpd.GeoDataFrame(fedex, geometry=gpd.points_from_xy(fedex.Longitude, fedex.Latitude))
fedex.crs = shp.crs


#Pulling in UPS data
ups = pd.read_excel(
    blob_folder+'MWPVL International Report on UPS North America Operations 2022.xlsb',
    sheet_name='1',
    engine='pyxlsb',
    header=3
    )[15:]

fix_cols = [str(i).replace('\n', ' ') for i in ups.columns]
ups.columns = fix_cols

ups = ups[ups['Country.1']=='USA']
# ups = ups[ups['Facility Open (Y/N)']=='Y']
# ups = ups[~ups['Facility Type (Primary Role)'].isin(['UPS Customer Center'])] #filter out retail stores
# ups = ups[~ups['Facility Type (Primary Role)'].isin(['UPS Freight Service Center'])] #filter out TL
ups = ups[~ups['Facility Type (Primary Role)'].isin(['UPS Supply Chain Solutions'])] #filter out dedicated 3PL
 
ups = gpd.GeoDataFrame(ups, geometry=gpd.points_from_xy(ups.Longitude, ups.Latitude))
ups.crs = shp.crs



### make sure these polygons are valid
if len(shp[shp.is_valid == False]) > 0:
        #shp_metro_bg =  gpd.GeoDataFrame(shp_metro_bg, geometry=shp_metro_bg.buffer(0))
    mask = shp.is_valid == False#['geometry']
    s_invalid = shp[mask]
    newval = gpd.GeoDataFrame(s_invalid, geometry=s_invalid.buffer(0))
    shp.loc[mask, 'geometry'] = newval['geometry']
    del(mask)
    del(newval)


if len(epa[epa.is_valid == False]) > 0:
        #shp_metro_bg =  gpd.GeoDataFrame(shp_metro_bg, geometry=shp_metro_bg.buffer(0))
    mask = epa.is_valid == False#['geometry']
    s_invalid = epa[mask]
    newval = gpd.GeoDataFrame(s_invalid, geometry=s_invalid.buffer(0))
    epa.loc[mask, 'geometry'] = newval['geometry']
    del(mask)
    del(newval)



##################################

csacbsa_list = centerpts[centerpts['use_csa'] == 1]['csatitle'].values.tolist()
csacbsa_list = csacbsa_list + centerpts[centerpts['use_csa'] == 0]['cbsa_metdiv_title'].values.tolist()

from shapely import wkt
import h3
from shapely.geometry import Polygon, mapping#, Point #, MultiPolygon #,Point, box, 
from shapely.ops import unary_union
# import geojson
# import geoplot as gplt
# import geoplot.crs as gcrs






def percent_rank(pd_series):
    return [(pd_series < value).astype(int).sum()/(len(pd_series) -1) for value in pd_series]

hex_locations = {}
infill = {}
metro_stock_cumulative = {}
metro_summary = {}
# max_yr = 2020


# from datetime import date
# from datetime import timedelta


centerpts_cbsas = centerpts['cbsacode'].values.tolist()
loop_metros = cbsa2fipsxw[cbsa2fipsxw['cbsacode'].isin(centerpts_cbsas)]['csa_cbsa_title'].drop_duplicates().values.tolist()
loop_metros.sort()

centerpts_count = centerpts[['csa_cbsa_title','lat_cbd','lon_cbd']].groupby(['csa_cbsa_title']).count()
max_cbd_n = centerpts_count['lon_cbd'].max()



airports_all = gpd.sjoin(airports, shp, how="inner", op="intersects") ##[airports['type'] == 'large_airport']
airports_count = airports_all[['csa_cbsa_title','longitude_deg']].groupby(['csa_cbsa_title']).count()
max_air_n = airports_count['longitude_deg'].max()




rail_intermodal_all = gpd.sjoin(rail_intermodal, shp, how="inner", op="intersects") 
rail_intermodal_count = rail_intermodal_all[['csa_cbsa_title','LON']].groupby(['csa_cbsa_title']).count()
max_rail_n = rail_intermodal_count['LON'].max()





epa_cutoff = 0.6667 ## top two quintiles = 0.6
airport_cutoff = 2
#infill_drivetime = 30
percentrank_cutoff = 0.7 ## for smaller metros we want a higher threshold, while for larger ones we want it lower

hexagon_resolution = 8
ring_miles = 25
dist_decay_alpha = 2 #1.655 #2 #1.75
dist_decay_beta = 0.2 #2 #1.75

dist_decay_cutoff = np.exp(-dist_decay_alpha * (ring_miles ** dist_decay_beta))

buffer_mi = 200


for l in loop_metros:
    
    print(l)
    shp_metro_bg = shp[shp['csa_cbsa_title'].isin([l])]
    shp_metro = shp_metro_bg.dissolve(by='csa_cbsa_title')
    shp_metro = gpd.GeoDataFrame(geometry=shp_metro['geometry'])
       
    # costar_metro = gpd.sjoin(costar_gdf, shp_metro, how="inner", op="intersects", lsuffix=None)
    # costar_metro = costar_metro.drop(columns='index_right')
    # costar_metro['n_properties'] = 1
        
    polygon_string = str(shp_metro['geometry'].iloc[0])
    shapely_polygon_fig = wkt.loads(polygon_string)
    polygons = shapely_polygon_fig
    boundary = unary_union(polygons)  # results in multipolygon sometimes
    boundary = boundary.convex_hull #MultiPolygon(Polygon(p.exterior) for p in boundary)
    boundary_metro = gpd.GeoDataFrame(geometry=gpd.GeoSeries(boundary))
    boundary_metro.crs = shp_metro.crs
    
    ## using a broader buffer for amazon, fedex, airports, etc
    boundary_metro_buffer = boundary_metro.to_crs(epsg=32710)
    boundary_metro_buffer = boundary_metro_buffer.geometry.buffer(1000*buffer_mi/mi_to_km)
    boundary_metro_buffer = gpd.GeoDataFrame(geometry=gpd.GeoSeries(boundary_metro_buffer))
    boundary_metro_buffer = boundary_metro_buffer.to_crs(shp_metro.crs)

    hexagons = h3.polyfill(mapping(boundary), res = hexagon_resolution, geo_json_conformant = True)
    polygonise = lambda hex_id: Polygon(
                                h3.h3_to_geo_boundary(
                                    hex_id, geo_json=True)
                                    )
    all_polys = gpd.GeoSeries((map(polygonise, hexagons)), \
                                      index=hexagons, \
                                      crs="EPSG:4326" \
                                     )
    ## h3 index in hex_gdf is reusable
    hex_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(all_polys))
    
    ## trying it without clipping, since clipping takes forever
    hex_gdf = gpd.sjoin(hex_gdf, shp_metro, how="inner", op="intersects", lsuffix=None)
    ##hex_gdf = gpd.clip(hex_gdf, mask=shp_metro)
    
    ## now make a shape from the csacbsa's geojson from osrm
    # infill_metro = gpd.GeoDataFrame(infill_df.merge(hex_gdf, right_index = True, left_on = 'hex_index'))
    
    hex_gdf = hex_gdf.to_crs(precincts.crs)
    hex_gdf['hex_index'] = hex_gdf.index
    
    ctrs = []
    for i in hex_gdf['hex_index'].to_list():
        ctr = h3.h3_to_geo(i)
        ctr = pd.DataFrame({'hex_index': [i], 'h3lat': [ctr[0]], 'h3lon': [ctr[1]]})
        ctrs.append(ctr)
    ctrs = pd.concat(ctrs)
    hex_gdf = hex_gdf.merge(ctrs, on='hex_index', how='left')
    
  


    
    shp_metro_bg = shp_metro_bg.to_crs(precincts.crs)
    
    epa_clipped = gpd.clip(epa, shp_metro) ## this takes a while
    epa_clipped = epa_clipped.to_crs(precincts.crs)
    
    epa_clipped['working_age_pop_time_decayed'] = epa_clipped[['d5ae', 'd5be']].max(axis=1) # greater of driving (d5ae) or transit (d5be)
    epa_clipped['working_age_pop_time_decayed_percentrank'] = percent_rank(epa_clipped['working_age_pop_time_decayed'])
    epa_clipped['jobs_time_decayed'] = epa_clipped[['d5ar', 'd5br']].max(axis=1) # greater of driving (d5ae) or transit (d5be)
    epa_clipped['jobs_time_decayed_percentrank'] = percent_rank(epa_clipped['jobs_time_decayed'])
    
    epa_clipped['d5ae_percentrank'] = percent_rank(epa_clipped['d5ae'])
    epa_clipped['d5ar_percentrank'] = percent_rank(epa_clipped['d5ar'])
    epa_clipped['d3aao_percentrank'] = percent_rank(epa_clipped['d3aao'])
    epa_clipped['d3a_percentrank'] = percent_rank(epa_clipped['d3a'])
    epa_clipped['d1c5_ind_percentrank'] = percent_rank(epa_clipped['d1c5_ind'])
       
    
    

    
    
       
    # cbre_clipped = gpd.clip(cbre, shp_metro) ## this takes a while
    # # if len(cbre_clipped) > 0:
    # #     cbre_clipped = cbre_clipped.to_crs(precincts.crs)
   
    # if len(cbre_clipped[cbre_clipped.is_valid == False]) > 0:
    #     mask = cbre_clipped.is_valid == False#['geometry']
    #     s_invalid = cbre_clipped[mask]
    #     newval = gpd.GeoDataFrame(s_invalid, geometry=s_invalid.buffer(0))
    #     cbre_clipped.loc[mask, 'geometry'] = newval['geometry']
    #     del(mask)
    #     del(newval)
   
    ### make sure these polygons are valid
    if len(shp_metro_bg[shp_metro_bg.is_valid == False]) > 0:
        mask = shp_metro_bg.is_valid == False#['geometry']
        s_invalid = shp[mask]
        newval = gpd.GeoDataFrame(s_invalid, geometry=s_invalid.buffer(0))
        shp_metro_bg.loc[mask, 'geometry'] = newval['geometry']
        del(mask)
        del(newval)

     ## for some reason, the precints.crs seems to make it invalid
    if len(epa_clipped[epa_clipped.is_valid == False]) > 0:
        mask = epa_clipped.is_valid == False#['geometry']
        s_invalid = epa_clipped[mask]
        newval = gpd.GeoDataFrame(s_invalid, geometry=s_invalid.buffer(0))
        epa_clipped.loc[mask, 'geometry'] = newval['geometry']
        del(mask)
        del(newval)    
        


    drivetime_metro = drivetime_poly[drivetime_poly['csa_cbsa_title'].isin([l])]
    drivetime_metro = drivetime_metro.dissolve(by='csa_cbsa_title')
    drivetime_metro = gpd.GeoDataFrame(geometry=drivetime_metro['geometry'])
       
        
    drivetime_metro = gpd.overlay(drivetime_metro, hex_gdf.to_crs(drivetime_metro.crs), how='intersection', make_valid=True)
    drivetime_metro = drivetime_metro.drop(columns='index_right')
    
  
    ## make one big infill hexgrid with interpolated data
    hex_gdf = hex_gdf.drop(columns='index_right')
    
    drivetime_hex = gpd.GeoDataFrame(hex_gdf)
    drivetime_hex['Infill_DriveTime'] = np.where(drivetime_hex['hex_index'].isin(drivetime_metro['hex_index'].to_list()), "Infill", "Outlying")
    
    drivetime_hex = drivetime_hex[['hex_index','Infill_DriveTime']]
    
    
    ## now let's calc the distances
    centerpts_metro = centerpts_gdf[centerpts_gdf['csa_cbsa_title'].isin([l])]
    ctr_list = centerpts_metro[['lat_cbd', 'lon_cbd']].values.tolist()
    h_list = ctrs[['hex_index', 'h3lat', 'h3lon']].values.tolist() #ctrs[['h3lat', 'h3lon']].values.tolist()
    
    ### now giving distance to each cbd
    dist_list = []
    for c in ctr_list:
        def ctrcalc(hh):
            return round(h3.point_dist(c, hh, unit='km') * mi_to_km, 3)
        # test = map(ctrcalc, h_list)
        temp_list = []
        for i in h_list:
            hhex = i[0]
            temp = ctrcalc(i[1:])
            temp = pd.DataFrame({'hex_index': [hhex], 'dist_cbd': [temp]})
            tempname = str('dist_cbd' + str(ctr_list.index(c)+1))
            temp = temp.rename(columns={'dist_cbd': tempname})
            temp_list.append(temp)
        temp_list = pd.concat(temp_list)
        dist_list.append(temp_list)
    
    ## need to join all these dfs
    if len(dist_list) > 1:
        from functools import reduce
        dist_df = reduce(lambda x, y: pd.merge(x, y, on = 'hex_index'), dist_list)
    else:
        dist_df = dist_list[0]

    ## dist_df = pd.concat(dist_list) ## this is more of an rbind
    ## dist_df['hex_index'] = dist_df.index
    dist_df.index = range(0, len(dist_df))
    ## now we have the min distance to a city center for every hex in the csa_cbsa
    
    
    ## if fewer cbd points than the max loop_metro, then add some nans to match where others are
    if len(ctr_list) < max_cbd_n:
        for i in range(1, max_cbd_n - len(ctr_list)+1):
            tempname = str('dist_cbd' + str(len(ctr_list)+i))
            dist_df[tempname] = float('nan')
    


    shp_metro_bg['hh_percentrank'] = percent_rank(shp_metro_bg['hh'])
    shp_metro_bg['med_hh_inc_percentrank'] = percent_rank(shp_metro_bg['med_hh_inc'])
    shp_metro_bg['median_owned_value_percentrank'] = percent_rank(shp_metro_bg['median_owned_value'])
    

    #shp_metro_bg
    hex_interpolated = area_interpolate(source_df=shp_metro_bg, target_df=hex_gdf, 
                                        allocate_total = True,
                                    extensive_variables=acs_extensive,
                                    intensive_variables=['hh_percentrank', 'med_hh_inc_percentrank', 'median_owned_value_percentrank'] 
#                                    extensive_variables=epa_extensive
                                # , categorical_variables=['hex_index']
                                # , spatial_index = 'target'
                                )
    
    
    hex_interpolated = hex_interpolated.merge(hex_gdf, on='geometry', how='left')
    hex_interpolated = hex_interpolated.merge(dist_df, on='hex_index', how='left')
    
    hex_interpolated = hex_interpolated.to_crs(shp.crs)
   
   
    hex_interpolated['dist_percentrank'] = percent_rank(-hex_interpolated['dist_cbd1']) 
    hex_interpolated['med_hh_inc'] = hex_interpolated['med_hh_inc_wt'] / hex_interpolated['hh']
    hex_interpolated['hh_ratio'] = (hex_interpolated['hh']+1)/(hex_interpolated['hh'].mean()+1)
    hex_interpolated['hh_logscore'] = np.log(hex_interpolated['hh_ratio'])
    
    hex_interpolated['hh_z'] = stats.zscore(hex_interpolated['hh'])
    hex_interpolated['hh_z_sqrt'] = stats.zscore(np.sqrt(hex_interpolated['hh']))
        
    hex_interpolated['hh_z_of_logscore'] = stats.zscore(hex_interpolated['hh_logscore'])
    
    
    epa_interpolated = area_interpolate(source_df=epa_clipped, target_df=hex_gdf, 
                                    extensive_variables=['ac_unpr', 'totemp','d1c5_ind'],
                                    intensive_variables=['d5ae','d5ar','d5be','d5br','d3aao','d3a','d3a_percentrank',
                                    'd1c5_ind_percentrank','working_age_pop_time_decayed','working_age_pop_time_decayed_percentrank',
                                    'jobs_time_decayed','jobs_time_decayed_percentrank',
                                    'd5ae_percentrank','d5ar_percentrank','d3aao_percentrank','natwalkind',
                                                         'noncom_vmt_per_worker','com_vmt_per_worker','vmt_per_worker']
#                                    extensive_variables=epa_extensive,
                                    # , categorical_variables=['hex_index']
                                   )
    epa_interpolated = epa_interpolated.merge(hex_gdf, on='geometry', how='left')
    epa_interpolated = epa_interpolated.to_crs(shp.crs)
    # epa_interpolated = gpd.clip(epa_interpolated, shp_metro) ## activate this if using EPA for maps
    # ## not adjusting here for potential na values
    
     ## adding 1 in this formula to avoid log issues
    epa_interpolated['d5ae_ratio'] = (epa_interpolated['d5ae']+1)/(epa_interpolated['d5ae'].mean()+1)
    epa_interpolated['d5ae_logscore'] = np.log(epa_interpolated['d5ae_ratio'])
   
    epa_interpolated['d5ae_z'] = stats.zscore(epa_interpolated['d5ae'])
    epa_interpolated['d5ae_z_sqrt'] = stats.zscore(np.sqrt(epa_interpolated['d5ae']))
       
    epa_interpolated['d5ae_z_of_logscore'] = stats.zscore(epa_interpolated['d5ae_logscore'])

    epa_interpolated['working_age_pop_time_decayed_ratio'] = (epa_interpolated['working_age_pop_time_decayed']+1)/(epa_interpolated['working_age_pop_time_decayed'].mean()+1)
    epa_interpolated['working_age_pop_time_decayed_logscore'] = np.log(epa_interpolated['working_age_pop_time_decayed_ratio'])
   
    epa_interpolated['working_age_pop_time_decayed_z'] = stats.zscore(epa_interpolated['working_age_pop_time_decayed'])
    epa_interpolated['working_age_pop_time_decayed_z_sqrt'] = stats.zscore(np.sqrt(epa_interpolated['working_age_pop_time_decayed']))
       
    epa_interpolated['working_age_pop_time_decayed_z_of_logscore'] = stats.zscore(epa_interpolated['working_age_pop_time_decayed_logscore'])

    
    ########################################################
   
    airports_metro = gpd.sjoin(airports, boundary_metro_buffer, how="inner", op="intersects")
    ports_metro = gpd.sjoin(ports, boundary_metro_buffer, how="inner", op="intersects")
    rail_intermodal_metro = gpd.sjoin(rail_intermodal, boundary_metro_buffer, how="inner", op="intersects")
    amazon_metro = gpd.sjoin(amazon, boundary_metro_buffer, how="inner", op="intersects")
    fedex_metro = gpd.sjoin(fedex, boundary_metro_buffer, how="inner", op="intersects")
    ups_metro = gpd.sjoin(ups, boundary_metro_buffer, how="inner", op="intersects")
    
    
    # airport_list = airports_metro[['latitude_deg', 'longitude_deg']].values.tolist()
    
    
    # dist_list = []
    # for c in airport_list:
    #     def ctrcalc(hh):
    #         return round(h3.point_dist(c, hh, unit='km') * mi_to_km, 3)
    #     # test = map(ctrcalc, h_list)
    #     temp_list = []
    #     for i in h_list:
    #         hhex = i[0]
    #         temp = ctrcalc(i[1:])
    #         temp = pd.DataFrame({'hex_index': [hhex], 'dist_airport': [temp]})
    #         temp_list.append(temp)
    #     temp_list = pd.concat(temp_list)
    #     dist_list.append(temp_list)
    
    # if len(dist_list) > 0:
    #     airport_dist_df = pd.concat(dist_list)
    # else:
    #     airport_dist_df = hex_gdf
    #     airport_dist_df['dist_airport'] = float('nan')
    #     airport_dist_df = airport_dist_df[['hex_index', 'dist_airport']]
        
    # airport_dist_df_min = airport_dist_df.groupby('hex_index').min()
    # airport_dist_df_min['hex_index'] = airport_dist_df_min.index
    # airport_dist_df_min.index = range(0, len(airport_dist_df_min))
    # airport_dist_df_min['Infill_Airport'] = np.where(airport_dist_df_min['dist_airport'] < airport_cutoff, "Airport_Proximate", "Airport_Outlying")
    
    airport_list = airports_metro[['latitude_deg', 'longitude_deg', 'EST_AREA', 'Feb2021_TTM_lb_millions']].values.tolist()
    dist_list = []
    for c in airport_list:
        def ctrcalc(hh):
            return round(h3.point_dist(c[0:2], hh, unit='km') * mi_to_km, 3)
        # test = map(ctrcalc, h_list)
        temp_list = []
        for i in h_list:
            hhex = i[0]
            temp = ctrcalc(i[1:])
            temp_sqft = c[2]
            temp_lb = c[3]
            decay_mult = np.exp(-dist_decay_alpha * (temp ** dist_decay_beta))
            air_sqft_index = temp_sqft * decay_mult
            air_lb_index = temp_lb * decay_mult
            temp = pd.DataFrame({'hex_index': [hhex], 'dist_airport': [temp], 'air_intermodal_sqft': [temp_sqft],
                                 'air_freight': [temp_lb],
                                 'air_intermodal_sqft_index': air_sqft_index,
                                 'air_freight_index': air_lb_index})
            temp_list.append(temp)
        temp_list = pd.concat(temp_list)
        dist_list.append(temp_list)
    
    if(len(dist_list) > 0):
        airport_dist_df = pd.concat(dist_list)
        airport_dist_df_min = airport_dist_df.groupby('hex_index').min()
        airport_dist_df_min['hex_index'] = airport_dist_df_min.index
        airport_dist_df_min.index = range(0, len(airport_dist_df_min))
        
        airport_dist_df_min = airport_dist_df_min[['hex_index', 'dist_airport']]
        
        airport_dist_df_index = airport_dist_df.groupby('hex_index').sum()
        airport_dist_df_index['hex_index'] = airport_dist_df_index.index
        airport_dist_df_index.index = range(0, len(airport_dist_df_index))
        airport_dist_df_index = airport_dist_df_index.drop(columns='dist_airport')
        airport_dist_df_index = airport_dist_df_index.merge(airport_dist_df_min, how='left', on='hex_index')
        airport_dist_df_index['Infill_Airport'] = np.where(airport_dist_df_index['dist_airport'] < airport_cutoff, "Airport_Proximate", "Airport_Outlying")
    else:
        airport_dist_df_index = pd.DataFrame()
   
    
    ports_metro['x'] = ports_metro['geometry'].apply(lambda p: p.x)
    ports_metro['y'] = ports_metro['geometry'].apply(lambda p: p.y)
    port_list = ports_metro[['y', 'x', 'IMPORTS']].values.tolist()
    dist_list = []
    for c in port_list:
        def ctrcalc(hh):
            return round(h3.point_dist(c[0:2], hh, unit='km') * mi_to_km, 3)
        # test = map(ctrcalc, h_list)
        temp_list = []
        for i in h_list:
            hhex = i[0]
            temp = ctrcalc(i[1:])
            temp_imports = c[2]
            decay_mult = np.exp(-dist_decay_alpha * (temp ** dist_decay_beta))
            sea_imports_index = temp_imports * decay_mult
            temp = pd.DataFrame({'hex_index': [hhex], 'dist_seaport': [temp], 'seaport_imports': [temp_imports],
                                 'sea_imports_index': sea_imports_index})
            temp_list.append(temp)
        temp_list = pd.concat(temp_list)
        dist_list.append(temp_list)
    
    if(len(dist_list) > 0):
        port_dist_df = pd.concat(dist_list)
        port_dist_df_min = port_dist_df.groupby('hex_index').min()
        port_dist_df_min['hex_index'] = port_dist_df_min.index
        port_dist_df_min.index = range(0, len(port_dist_df_min))
        
        port_dist_df_min =port_dist_df_min[['hex_index', 'dist_seaport']]
        
        port_dist_df_index = port_dist_df.groupby('hex_index').sum()
        port_dist_df_index['hex_index'] = port_dist_df_index.index
        port_dist_df_index.index = range(0, len(port_dist_df_index))
        port_dist_df_index= port_dist_df_index.drop(columns='dist_seaport')
        port_dist_df_index= port_dist_df_index.merge(port_dist_df_min, how='left', on='hex_index')
        port_dist_df_index['Infill_Seaport'] = np.where(port_dist_df_index['dist_seaport'] < airport_cutoff, "Seaport_Proximate", "Seaport_Outlying")
    else:
        port_dist_df_index = pd.DataFrame()
        
   
    
    rail_intermodal_metro['N'] = 1
    rail_intermodal_list = rail_intermodal_metro[['LAT', 'LON', 'N']].values.tolist()
        
    dist_list = []
    for c in rail_intermodal_list:
        def ctrcalc(hh):
            return round(h3.point_dist(c[0:2], hh, unit='km') * mi_to_km, 3)
        # test = map(ctrcalc, h_list)
        temp_list = []
        for i in h_list:
            hhex = i[0]
            temp = ctrcalc(i[1:])
            temp_n = c[2]
            decay_mult = np.exp(-dist_decay_alpha * (temp ** dist_decay_beta))
            rail_intermodal_index = temp_n * decay_mult
            temp = pd.DataFrame({'hex_index': [hhex], 'dist_rail_intermodal': [temp], #'rail_intermodal_n': [temp_sqft],
                                 'rail_intermodal_index': rail_intermodal_index
                                 })
            temp_list.append(temp)
        temp_list = pd.concat(temp_list)
        dist_list.append(temp_list)
    
    if(len(dist_list) > 0):
        rail_intermodal_dist_df = pd.concat(dist_list)
        rail_intermodal_dist_df_min = rail_intermodal_dist_df.groupby('hex_index').min()
        rail_intermodal_dist_df_min['hex_index'] = rail_intermodal_dist_df_min.index
        rail_intermodal_dist_df_min.index = range(0, len(rail_intermodal_dist_df_min))
        
        rail_intermodal_dist_df_min = rail_intermodal_dist_df_min[['hex_index', 'dist_rail_intermodal']]
        
        rail_intermodal_dist_df_index = rail_intermodal_dist_df.groupby('hex_index').sum()
        rail_intermodal_dist_df_index['hex_index'] = rail_intermodal_dist_df_index.index
        rail_intermodal_dist_df_index.index = range(0, len(rail_intermodal_dist_df_index))
        rail_intermodal_dist_df_index= rail_intermodal_dist_df_index.drop(columns='dist_rail_intermodal')
        rail_intermodal_dist_df_index= rail_intermodal_dist_df_index.merge(rail_intermodal_dist_df_min, how='left', on='hex_index')
        rail_intermodal_dist_df_index['Infill_Rail_Intermodal'] = np.where(rail_intermodal_dist_df_index['dist_rail_intermodal'] < airport_cutoff, "Rail_Intermodal_Proximate", "Rail_Intermodal_Outlying")
    else:
        rail_intermodal_dist_df_index = pd.DataFrame()
    
    
    
    
    
    
    
   
    
   
    amazon_list = amazon_metro[['Latitude', 'Longitude', 'Total Working Square  Feet', 'Annual Packages Shipped']].values.tolist()
    
    
    dist_list = []
    for c in amazon_list:
        def ctrcalc(hh):
            return round(h3.point_dist(c[0:2], hh, unit='km') * mi_to_km, 3)
        # test = map(ctrcalc, h_list)
        temp_list = []
        for i in h_list:
            hhex = i[0]
            temp = ctrcalc(i[1:])
            temp_sqft = c[2]
            temp_pkg = c[3]
            decay_mult = np.exp(-dist_decay_alpha * (temp ** dist_decay_beta))
            amazon_sqft_index = temp_sqft * decay_mult
            amazon_packages_index = temp_pkg * decay_mult
            temp = pd.DataFrame({'hex_index': [hhex], 'dist_amazon': [temp], 'amazon_sqft': [temp_sqft],
                                 'amazon_packages': [temp_pkg], 'amazon_sqft_index': amazon_sqft_index,
                                 'amazon_packages_index': amazon_packages_index})
            temp_list.append(temp)
        temp_list = pd.concat(temp_list)
        dist_list.append(temp_list)
    
    amazon_dist_df = pd.concat(dist_list)
    amazon_dist_df_min = amazon_dist_df.groupby('hex_index').min()
    amazon_dist_df_min['hex_index'] = amazon_dist_df_min.index
    amazon_dist_df_min.index = range(0, len(amazon_dist_df_min))
    
    amazon_dist_df_index = amazon_dist_df.groupby('hex_index').sum()
    amazon_dist_df_index['hex_index'] = amazon_dist_df_index.index
    amazon_dist_df_index.index = range(0, len(amazon_dist_df_index))
    
    

    fedex_list = fedex_metro[['Latitude', 'Longitude', 'Facility Square  Feet', 'Facility Type (Primary Role)']].values.tolist()
        
    dist_list = []
    for c in fedex_list:
        def ctrcalc(hh):
            return round(h3.point_dist(c[0:2], hh, unit='km') * mi_to_km, 3)
        # test = map(ctrcalc, h_list)
        temp_list = []
        for i in h_list:
            hhex = i[0]
            temp = ctrcalc(i[1:])
            temp_sqft = c[2]
            decay_mult = np.exp(-dist_decay_alpha * (temp ** dist_decay_beta))
            fedex_sqft_index = temp_sqft * decay_mult
            temp = pd.DataFrame({'hex_index': [hhex], 'dist_fedex': [temp], 'fedex_sqft': [temp_sqft],
                                 'fedex_sqft_index': fedex_sqft_index, 'facility_type':c[3]
                                 })
            temp_list.append(temp)
        temp_list = pd.concat(temp_list)
        dist_list.append(temp_list)
    
    fedex_dist_df = pd.concat(dist_list)
    fedex_dist_df_min = fedex_dist_df[~fedex_dist_df['facility_type'].isin(['FedEx Ship Center'])]
    
    fedex_dist_df_min = fedex_dist_df_min.groupby('hex_index').min()
    fedex_dist_df_min['hex_index'] = fedex_dist_df_min.index
    fedex_dist_df_min.index = range(0, len(fedex_dist_df_min))
    
    fedex_dist_df_index = fedex_dist_df.groupby('hex_index').sum()
    fedex_dist_df_index['hex_index'] = fedex_dist_df_index.index
    fedex_dist_df_index.index = range(0, len(fedex_dist_df_index))
    
    
    
    
    
    ups_list = ups_metro[['Latitude', 'Longitude', 'Facility Square  Feet', 'Facility Type (Primary Role)']].values.tolist()
        
    dist_list = []
    for c in ups_list:
        def ctrcalc(hh):
            return round(h3.point_dist(c[0:2], hh, unit='km') * mi_to_km, 3)
        # test = map(ctrcalc, h_list)
        temp_list = []
        for i in h_list:
            hhex = i[0]
            temp = ctrcalc(i[1:])
            temp_sqft = c[2]
            decay_mult = np.exp(-dist_decay_alpha * (temp ** dist_decay_beta))
            ups_sqft_index = temp_sqft * decay_mult
            temp = pd.DataFrame({'hex_index': [hhex], 'dist_ups': [temp], 'ups_sqft': [temp_sqft],
                                 'ups_sqft_index': ups_sqft_index, 'facility_type':c[3]
                                 })
            temp_list.append(temp)
        temp_list = pd.concat(temp_list)
        dist_list.append(temp_list)
    
    ups_dist_df = pd.concat(dist_list)
    ups_dist_df_min = ups_dist_df[~ups_dist_df['facility_type'].isin(['UPS Customer Center'])]
    
    ups_dist_df_min = ups_dist_df_min.groupby('hex_index').min()
    ups_dist_df_min['hex_index'] = ups_dist_df_min.index
    ups_dist_df_min.index = range(0, len(ups_dist_df_min))
    
    ups_dist_df_index = ups_dist_df.groupby('hex_index').sum()
    ups_dist_df_index['hex_index'] = ups_dist_df_index.index
    ups_dist_df_index.index = range(0, len(ups_dist_df_index))
    
    
    
    ## this method takes too much memory. maybe better to select h3s within a range and then calc and then reassemble
    
    # from scipy.spatial import distance_matrix
    # hex_dist = pd.DataFrame(hex_interpolated, columns=['h3lon', 'h3lat'], index=hex_interpolated['hex_index'])
    # hex_dist_result = pd.DataFrame(distance_matrix(hex_dist.values, hex_dist.values), index=hex_dist.index, columns=hex_dist.index)
    # hex_dist_result = 1 / (hex_dist_result ** dist_decay_exp)
    # ## now theoretically we can multiply this by any of the metrics to get a proximity-weighted metric
    
    
    ### Question: Does being near an airport or port mean a location is NOT infill? 
   
    
    # infill_poly = gpd.overlay(union, hex_interpolated, how='intersection', make_valid=True)
    # infill_poly['csacbsa'] = l
    # infill_poly = infill_poly.dissolve(by='csacbsa')
    # infill_poly = gpd.GeoDataFrame(geometry=infill_poly['geometry'])
    
    ## can't figure out a good way to interpolate these right now -- won't use hexagons
    ## will just use center points for now
    hex_gdf_points = hex_gdf.drop(columns='geometry') 
    hex_gdf_points = gpd.GeoDataFrame(hex_gdf_points, geometry=gpd.points_from_xy(hex_gdf_points.h3lon, hex_gdf_points.h3lat))
    
    
    
    # hex_gdf_points.crs = cbre_clipped.crs
    # # airports_metro = gpd.sjoin(airports[airports['type'] == 'large_airport'], shp_metro, how="inner", op="intersects")
    
    # cbre_metro = gpd.sjoin(hex_gdf_points.to_crs(shp.crs), cbre.to_crs(shp.crs), how="left", op="intersects")
    # cbre_metro = cbre_metro[['hex_index', 'MktName', 'MktCode', 'MktCode6', 
    #                           'SubMktName', 'SubMktCode', 'ActiveSub',
    #                           'Tier', 'InSumofMkt']]
  
    # ## make one big infill hexgrid with interpolated data
    if 'index_right' in hex_interpolated.columns:    
        hex_interpolated = hex_interpolated.drop(columns='index_right')
    
  

    

    data_df = hex_interpolated.merge(epa_interpolated, on="hex_index", how='left')
    if 'geometry_y' in data_df.columns:    
        data_df = data_df.drop(columns='geometry_y')
    data_df = data_df.rename(columns={'geometry_x': 'geometry'})
    
    # data_df = data_df.merge(cbre_metro, on="hex_index", how='left')
    data_df = data_df.merge(drivetime_hex, on="hex_index", how='left')
    
    if len(airport_dist_df_index) > 0:
        data_df = data_df.merge(airport_dist_df_index, on="hex_index", how='left')
    
    if not 'dist_airport' in data_df.columns:
        data_df[['air_intermodal_sqft', 'air_freight', 
       'air_intermodal_sqft_index', 'air_freight_index', 
       'dist_airport', 'Infill_Airport']] = np.nan
    
    if len(port_dist_df_index) > 0:
        data_df = data_df.merge(port_dist_df_index, on="hex_index", how='left')
    
    if len(rail_intermodal_dist_df_index) > 0:
        data_df = data_df.merge(rail_intermodal_dist_df_index, on="hex_index", how='left')
    
    if not 'dist_seaport' in data_df.columns:
        data_df[['dist_seaport', 'seaport_imports',
                                  'sea_imports_index', 'Infill_Seaport']] = np.nan
    if not 'dist_rail_intermodal' in data_df.columns:
        data_df[['dist_rail_intermodal', 
                                  'rail_intermodal_index',  'Infill_Rail_Intermodal']] = np.nan
    
    data_df = data_df.merge(amazon_dist_df_min[['hex_index','dist_amazon']], on="hex_index", how='left')
    data_df = data_df.merge(amazon_dist_df_index[['hex_index','amazon_sqft_index', 'amazon_packages_index']], on="hex_index", how='left')
    data_df = data_df.merge(fedex_dist_df_min[['hex_index','dist_fedex']], on="hex_index", how='left')
    data_df = data_df.merge(fedex_dist_df_index[['hex_index','fedex_sqft_index']], on="hex_index", how='left')
    data_df = data_df.merge(ups_dist_df_min[['hex_index','dist_ups']], on="hex_index", how='left')
    data_df = data_df.merge(ups_dist_df_index[['hex_index','ups_sqft_index']], on="hex_index", how='left')
   
    
    data_df['Infill_EPA'] = np.where(data_df['working_age_pop_time_decayed_percentrank'] > epa_cutoff, "Infill", "Outlying")
    
    
    
    
    data_df['Infill'] = np.where((data_df['Infill_EPA'] == "Infill") | (data_df['Infill_DriveTime'] == "Infill"), "Infill", "Outlying")
    data_df['Trade_Proximate'] = np.where((data_df['Infill_Airport'] == "Airport_Proximate") | (data_df['Infill_Seaport'] == "Seaport_Proximate") | (data_df['Infill_Rail_Intermodal'] == "Rail_Intermodal_Proximate"), "Trade_Proximate", "Outlying")
    data_df['Infill_Combined'] = np.where((data_df['Trade_Proximate'] == "Trade_Proximate"), "Trade_Proximate", np.where((data_df['Infill_EPA'] == "Infill") | (data_df['Infill_DriveTime'] == "Infill"), "Infill", "Outlying"))
   
   ### ADD UPS / FEDEX / AMAZON PERCENTRANKS 
    data_df['ups_sqft_index_pctrank'] = percent_rank(-data_df['ups_sqft_index']) 
    data_df['fedex_sqft_index_pctrank'] = percent_rank(-data_df['fedex_sqft_index']) 
    data_df['amazon_packages_index_pctrank'] = percent_rank(-data_df['amazon_packages_index']) 
   
    # data_df['Infill'] = data_df['Infill_EPA']
    # data_df['Infill'] = np.where((data_df['Infill_EPA'] == "Infill") | (data_df['Infill_Airport'] == "Infill"), "Infill", "Outlying")
    
    data_df['activity_density'] = (data_df['totemp'] + data_df['hh']) / data_df['ac_unpr']    
    data_df['com_vmt_per_worker_pctrank'] = percent_rank(-data_df['com_vmt_per_worker']) 
    data_df['noncom_vmt_per_worker_pctrank'] = percent_rank(-data_df['noncom_vmt_per_worker']) 
    # data_df['dist_amazon_percentrank'] = percent_rank(-data_df['dist_amazon']) 
    data_df['homeownership'] = data_df['owner_hu'] / data_df['tot_occ_hu']
    data_df['poverty_share'] = data_df['poverty_hh'] / data_df['poverty_hh_denom']
    data_df['college_plus_sh'] = data_df['college_plus'] / data_df['edu_denom']
    data_df['masters_plus_sh'] = data_df['masters_plus'] / data_df['edu_denom']
    data_df['ba_masters_blend_sh'] = 0.75 * data_df['college_plus_sh'] + 0.25 * data_df['masters_plus_sh']
    data_df['median_owned_value'] = data_df['median_owned_value_wt'] / data_df['owner_hu']
    data_df['med_hh_inc'] = data_df['med_hh_inc_wt'] / data_df['hh']
    data_df['neighborhood_median_yr_built'] = data_df['neighborhood_median_yr_built_wt'] / data_df['neighborhood_median_yr_built_denom']
    data_df = data_df.rename(columns={'index_right': 'csa_cbsa_title'})
    data_df['csa_cbsa_title'] = l
    
    if 'h3lon_y' in data_df.columns:    
        data_df = data_df.drop(columns=['h3lon_y', 'h3lat_y'])
        data_df = data_df.rename(columns={'h3lon_x': 'h3lon', 'h3lat_x' : 'h3lat'})
    
    ## export the location data for rent prediction in R
    csacbsa = l.translate ({ord(c): "_" for c in "!@#$%^&*()[]{};:,./<>?\|`~=+ "}) ## keeping - 
    
    save_folder = str(blob_folder +'Hex_Locations_'+format(csacbsa)+'.csv')
    data_df.to_csv(save_folder)
    
    
    infill_df = data_df[['hex_index', 'Infill', 'Trade_Proximate']]
    data_df = gpd.GeoDataFrame(data_df, geometry=data_df['geometry'], crs = shp_metro.crs)
    
    
    
    
    
    
    # infill_df = infill_df[infill_df['Infill'] == "Infill"]
    infill_gdf = hex_gdf.merge(infill_df, on="hex_index", how="inner")
    infill_gdf = infill_gdf.dissolve(by='Infill')
    
    
    infill_only_gdf = infill_gdf[infill_gdf.index == "Infill"]
    infill_shp = gpd.GeoDataFrame(infill_only_gdf, geometry=infill_only_gdf["geometry"], crs=infill_gdf.crs)
    infill_shp = infill_shp.to_crs(shp_metro.crs)
    
   

    
    ax = infill_shp.plot(figsize=(10, 10), alpha=0.5, edgecolor='k') # edgecolor='k'
    shp_metro.geometry.boundary.plot(color=None,edgecolor='k',linewidth = 2,ax=ax) #Use your second dataframe
    centerpts_metro.plot(ax=ax, marker = '*', color='black', markersize=10)
    ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite, crs='EPSG:4326')
    ax.set_axis_off()

    ### let's save a picture of this
    save_folder = str(blob_folder +'Images_Infill_Map_'+format(csacbsa)+'.png')
    #ax.savefig(save_folder)
    ax.figure.savefig(save_folder, dpi=300)

    # Basic choropleth
    """map_shp = data_df
    map_shp['color_col'] = np.log(map_shp['ups_sqft_index'])
    ax = map_shp.plot(column = 'color_col', cmap = "inferno", figsize=(20, 20), alpha=0.8, edgecolor=None, linewidth=0.001) # edgecolor='k'
    shp_metro.geometry.boundary.plot(color=None,edgecolor='k',linewidth = 2,ax=ax) #Use your second dataframe
    centerpts_metro.plot(ax=ax, marker = '*', color='black', markersize=10)
    ctx.add_basemap(ax, source=ctx.providers.Stamen.TonerLite, crs='EPSG:4326')
    ax.set_axis_off()"""

    """infill[l] = infill_df
    hex_locations[l] = data_df"""


#################################################################################
#################################################################################

#### remember that pd.concat needs the same cols
## need to have dist_cbd2 ... dist_cbdn cols

"""
infill_df = pd.concat(infill)
infill_df.index = range(len(infill_df))
save_folder = str(blob_folder+ 'PythonOut_infill_hex_index.csv')
infill_df.to_csv(save_folder)


hex_locations_df = pd.concat(hex_locations)
hex_locations_df.index = range(len(hex_locations_df))
save_folder = str(blob_folder+ 'PythonOut_all_hex_locations.csv')
hex_locations_df.to_csv(save_folder)
"""
