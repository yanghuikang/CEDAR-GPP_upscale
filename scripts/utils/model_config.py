# fill value
FILL_VALUE = -9999
SCALE_FACTOR = 0.01

# default random seed
RANDOM_STATE = 18

# 2001 baseline CO2 concentration
co2_2001 = 370.09625

# list of C4 and other sites to remove from the training data
rm_site_list = ['AU-Stp','AU-TTE','PA-SPs','US-IB2',
            'US-LWW','US-Wkg','US-AR1','US-AR2', 'US-ARb','US-ARc','US-Bi2','US-Ro4',
            'US-SRG','US-Tw2','IT-BCi']

# dictionaries to rename variables
era5_dict = {'temperature_2m':'Tmean','potential_evaporation':'PET','skin_temperature':'Ts',
            'evaporation_from_vegetation_transpiration':'transpiration',
            'dewpoint_temperature_2m':'dewpoint','total_precipitation':'prcp',
            'surface_solar_radiation_downwards':'RSDN'}


# dictionaries for MODIS Land cover to PFT
# MODIS land cover to PFT and onehot
modis_dict = {1:'ENF',2:'EBF',3:'DNF',4:'DBF',5:'MF',6:'CSH',7:'OSH',8:'WSA',
              9:'SAV',10:'GRA',11:'WET',12:'CRO',13:'URB',14:'CVM',15:'SNO',
              16:'BSV',17:'WAT'}
modis_pft_dict = {1:'ENF',2:'EBF',3:'DNF',4:'DBF',5:'MF',6:'SH',7:'SH',8:'SA',
              9:'SA',10:'GRA',11:'Other',12:'CRO',13:'Other',14:'Other',15:'Other',
              16:'Other',17:'Other'}

# dictionary for Whittaker biomes
wt_dict = {1:'Tropical seasonal forest/savanna',2:'Subtropical desert',3:'Temperate rain forest',
    4:'Tropical rain forest',5:'Woodland/shrubland',6:'Tundra',7:'Boreal forest',8:'Temperate grassland/desert',
    9:'Temperate seasonal forest'}

wt_cat_order = ['Tropical rain forest','Tropical seasonal forest/savanna','Subtropical desert',
    'Temperate rain forest','Temperate seasonal forest','Woodland/shrubland',
    'Temperate grassland/desert','Boreal forest','Tundra']

# dictionaries for grouping IGBP categories to broader PFT
pft_dict = {'CSH':'SH','OSH':'SH','WSA':"SA","SAV":"SA"}

# mapping koppen code to level 1 regions
koppen_dict = {1:'Tropical',2:'Tropical',3:'Tropical',4:'Arid',5:'Arid',6:'Arid',7:'Arid',
        8:'Temperate',9:'Temperate',10:'Temperate',11:'Temperate',12:'Temperate',13:'Temperate',
        14:'Temperate',15:'Temperate',16:'Temperate',17:'Cold',18:'Cold',19:'Cold',20:'Cold',
        21:'Cold',22:'Cold',23:'Cold',24:'Cold',25:'Cold',26:'Cold',27:'Cold',28:'Cold',
        29:'Polar',30:'Polar'}


# dictionary of datasets and their variables for model deployment
dataset_shortterm_dict = {
    # 'ALEXI':['ET'],  # y coordiante is reversed (starts from negative values)
    'BESS_Rad':['BESS_PAR','BESS_PARdiff','BESS_RSDN'],
    'CSIF':['CSIF-SIFdaily','CSIF-SIFinst'],  # y coordiante is reversed
    'ERA5':['p1','p2'],
    'ESA_CCI':['ESACCI-sm'],
    'MCD12Q1':['IGBP_mode'],
    'MCD43C4v006':['b1','b2','b3','b4','b5','b6','b7','EVI','GCI','NDVI','NDWI','NIRv','kNDVI','Percent_Snow'],
    'MODIS_LAI':['Fpar','Lai'],
    'MODIS_LST':['LST_Day','LST_Night'],
    'koppen':['koppen']}

# current version
dataset_longterm_dict = {        
    'ERA5':['p1','p2'],
    # 'ESA_CCI':['ESACCI-sm'],
    'PKU_GIMMS_NDVI_V1.0_005':['NDVI'],
    'MCD12Q1':['IGBP_mode'],
    # 'PKU_GIMMS_LAI4g_202211_005':['LAI'],
    'PKU_GIMMS_LAI4g_V1.0_005':['LAI'],
    'koppen':['koppen']}


# List of features
# ET removed from the training set
ft_st_model0 = ['BESS-PAR', 'BESS-PARdiff', 'BESS-RSDN', 'PET', 'Ts', 'Tmean', 'prcp', 'vpd', 'prcp-lag3',
            'ESACCI-sm', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'EVI', 'NDVI', 'GCI', 'NDWI', 
            'NIRv', 'kNDVI', 'CSIF-SIFdaily', 'Percent_Snow', 'Fpar', 'Lai', 'LST_Day', 'LST_Night', 
            'MODIS_PFT_CRO', 'MODIS_PFT_DBF', 'MODIS_PFT_EBF', 'MODIS_PFT_ENF', 'MODIS_PFT_GRA', 
            'MODIS_PFT_MF', 'MODIS_PFT_Other', 'MODIS_PFT_SA', 'MODIS_PFT_SH','koppen_Arid', 'koppen_Cold', 
            'koppen_Polar', 'koppen_Temperate','koppen_Tropical']

ft_st_model1 = ['BESS-PAR', 'BESS-PARdiff', 'BESS-RSDN', 'PET', 'Ts', 'Tmean', 'prcp', 'vpd', 'prcp-lag3',
            'ESACCI-sm', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'EVI', 'NDVI', 'GCI', 'NDWI', 
            'NIRv', 'kNDVI', 'CSIF-SIFdaily', 'Percent_Snow', 'Fpar', 'Lai', 'LST_Day', 'LST_Night', 
            'MODIS_PFT_CRO', 'MODIS_PFT_DBF', 'MODIS_PFT_EBF', 'MODIS_PFT_ENF', 'MODIS_PFT_GRA', 
            'MODIS_PFT_MF', 'MODIS_PFT_Other', 'MODIS_PFT_SA', 'MODIS_PFT_SH','koppen_Arid', 'koppen_Cold', 
            'koppen_Polar', 'koppen_Temperate','koppen_Tropical','year']

ft_st_model2 = ['BESS-PAR', 'BESS-PARdiff', 'BESS-RSDN', 'PET', 'Ts', 'Tmean', 'prcp', 'vpd', 'prcp-lag3',
            'ESACCI-sm', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'EVI', 'NDVI', 'GCI', 'NDWI', 
            'NIRv', 'kNDVI', 'CSIF-SIFdaily', 'Percent_Snow', 'Fpar', 'Lai', 'LST_Day', 'LST_Night', 
            'MODIS_PFT_CRO', 'MODIS_PFT_DBF', 'MODIS_PFT_EBF', 'MODIS_PFT_ENF', 'MODIS_PFT_GRA', 
            'MODIS_PFT_MF', 'MODIS_PFT_Other', 'MODIS_PFT_SA', 'MODIS_PFT_SH','koppen_Arid', 'koppen_Cold', 
            'koppen_Polar', 'koppen_Temperate','koppen_Tropical','CO2_concentration']

# update 02/02/2023 add NDVI
ft_lt_model0 = ['PET','Ts','Tmean','prcp','vpd','prcp-lag3','RSDN','LAI','NDVI','MODIS_PFT_CRO', 'MODIS_PFT_DBF', 
                'MODIS_PFT_EBF', 'MODIS_PFT_ENF', 'MODIS_PFT_GRA', 'MODIS_PFT_MF', 'MODIS_PFT_Other', 
                'MODIS_PFT_SA', 'MODIS_PFT_SH','koppen_Arid', 'koppen_Cold', 'koppen_Polar', 'koppen_Temperate',
                'koppen_Tropical']

ft_lt_model1 = ft_lt_model0 + ['year']

ft_lt_model2 = ft_lt_model0 + ['CO2_concentration']

ft_dict = {'ST-model0':ft_st_model0, 'ST-model1':ft_st_model1, 'ST-model2':ft_st_model2,
           'LT-model0':ft_lt_model0, 'LT-model1':ft_lt_model1, 'LT-model2':ft_lt_model2}

ft_setup_dict = {'ST_Baseline':ft_st_model0, 'ST_CFE':ft_st_model2, 'LT_Baseline': ft_lt_model0, 'LT_CFE': ft_lt_model2}

# Optimal hyper parameters for each model scenario
param_st_dict = {'xgboost':{'colsample_bytree': 0.3, 'gamma': 0, 'learning_rate': 0.01, 'verbosity':0, 'n_estimators': 500},
                 'lasso':{'alpha': 0.01},
                 'svr':{'gamma': 'auto', 'C': 1},
                 'rf':{'n_estimators': 500, 'min_samples_split': 16, 'min_samples_leaf': 10, 'max_depth': 15.0},
                 'nn':{'learning_rate_init': 1e-05, 'hidden_layer_sizes': (64, 64, 64), 'batch_size': 128}
                }


param_lt_dict = {'xgboost':{'colsample_bytree': 0.3, 'gamma': 0, 'learning_rate': 0.05, 'verbosity':0, 'n_estimators': 300},
                 'lasso':{'alpha':0.01},
                 'svr':{'gamma': 'auto', 'C': 1},
                 'rf':{'n_estimators': 500, 'min_samples_split': 12, 'min_samples_leaf': 5, 'max_depth': 15.0},
                 'nn':{'learning_rate_init': 1e-05, 'hidden_layer_sizes': (64, 64, 64), 'batch_size': 128}
                }
param_dict = {'ST':param_st_dict,'LT':param_lt_dict}
