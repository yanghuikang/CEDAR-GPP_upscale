import numpy as np
import pandas as pd
import time
from dask_ml.preprocessing import DummyEncoder
from dask.distributed import wait

import xarray as xr
from pandas.tseries.offsets import MonthEnd

import xgboost as xgb
xgb.config_context(verbosity=0)

import sys
sys.path.insert(0, '../utility')
import os
import model_config as mcf
import cfe as cfe
import model_deployment as md

## UPDATES
# 02/28/2023
#   Changed "SCALE_FACTOR"

def mergeDataset(year, month, dataset_dict, verbose=False):
    """
    Merge input/feature datasets as a single Xarray dataset
    for: model deployment

    """
    
    zarr_path = '/global/scratch/users/yanghuikang/upscale/data/processed/monthly/'
    ds = xr.Dataset()
    
    for dataset_name in dataset_dict.keys():

        if verbose:
            print(dataset_name)

        var_list = dataset_dict[dataset_name]

        for var_name in var_list:

            if dataset_name == 'koppen':
                zarr_file = zarr_path+dataset_name+'/zarr_yearly/'+var_name
            elif dataset_name == 'MCD12Q1':
                zarr_file = zarr_path+dataset_name+'/zarr_yearly/'+var_name
            else:
                zarr_file = zarr_path+dataset_name+'/zarr_yearly/'+var_name + '/' + str(year)

            with xr.open_zarr(zarr_file,consolidated=False) as ds_single:
                
                # select one month of data
                if dataset_name not in ['MCD12Q1','koppen']:
                    ds_single = ds_single.isel(time=(ds_single.time.dt.month==month))
                else:
                    time_index = pd.to_datetime([str(year)+str(month).zfill(2)], format="%Y%m") + MonthEnd(1)
                    ds_single = ds_single.expand_dims(dim={'time':time_index})

                # round up the coordiantes to align them
                ds_single['x'] = [k.round(3) for k in ds_single.x.values]
                ds_single['y'] = [k.round(3) for k in ds_single.y.values]
                # print(ds_single['y'].values)

                # rename ERA5 bands
                col_dict = {key:value for (key,value) in mcf.era5_dict.items() if key in ds_single.keys()}
                if dataset_name == 'ERA5':
                    ds_single = ds_single.rename(col_dict)

                # remove QA bands
                if 'QA' in list(ds_single.keys()):
                    # ds_single = ds_single.rename({'QA':var_name+'_QA'})
                    ds_single = ds_single.drop_vars(['QA'])
                                                
                # drop spatial_ref
                if 'spatial_ref' in ds_single.dims:
                    ds_single = ds_single.drop_dims(['spatial_ref'])
                
                ds = xr.merge([ds,ds_single])
  
    return ds


# prepare dataframe for model deployment
def prepareDataframe(model_name, year, month):
    """
    Prepare Dask Dataframe for 
    """
    
    if 'ST' in model_name: # short term model
        dataset_list = mcf.dataset_shortterm_dict
    elif 'LT' in model_name:
        dataset_list = mcf.dataset_longterm_dict
    else:
        print('error: check model name', flush=True)
    
    # get merged spatial datasets
    # print('loading zarr datasets ...',flush=True)
    start = time.time()
    ds = md.mergeDataset(year, month, dataset_list)
    ds = ds.squeeze(dim='time',drop=True) # remove time dimension
    ds = ds.drop_vars(['spatial_ref'])
    print('load zarr datasets took {} seconds'.format(int(time.time()-start)), flush=True)

    # get feature list
    # model_type: NT_Baseline, NT_CFE, DT_Baseline, DT_CFE
    model_type = '_'.join(model_name.split('_')[0:2])
    if '-' in model_type:
        model_type = model_type.split('-')[0]
        
    ft_list = mcf.ft_setup_dict[model_type]
    input_vars = [x for x in ft_list if ('MODIS_PFT_' not in x) & ('koppen_' not in x) & ('CO2_' not in x)] # remove dummy vars
    input_vars = input_vars + ['IGBP','koppen'] # add categorical vars
    ds = ds[input_vars]
    
    # convert dataset to dask dataframe and persist the data
    start = time.time()
    if 'LT' in model_name:
        ds = ds.chunk(chunks={'x':500,'y':500})
        print('chunking...')
    df = ds.unify_chunks().to_dask_dataframe(dim_order = ['y','x'])
    df_persist = df.persist()
    wait([df_persist])
    print('convert to dask dataframe took {} seconds'.format(int(time.time()-start)), flush=True)
    start = time.time()
    
    # create One Hot encoding for PFT
    # map modis IGBP to PFT (grouped categories)
    modis_pft_dict = {1:'ENF',2:'EBF',3:'DNF',4:'DBF',5:'MF',6:'SH',7:'SH',8:'SA',
                  9:'SA',10:'GRA',11:'Other',12:'CRO',13:'Other',14:'Other',15:'Other',
                  16:'Other',17:'Other'}
    df_persist['MODIS_PFT'] = df_persist['IGBP'].map(modis_pft_dict)

    # convert the PFT column to category (necessary for dummy encoder)
    df_persist = df_persist.categorize(columns=['MODIS_PFT'])

    # create one hot encoding
    enc = DummyEncoder(columns=['MODIS_PFT'])
    df_persist = enc.fit_transform(df_persist)

    # create One Hot encoding for Koppen
    koppen_dict = {1:'Tropical',2:'Tropical',3:'Tropical',4:'Arid',5:'Arid',6:'Arid',7:'Arid',
        8:'Temperate',9:'Temperate',10:'Temperate',11:'Temperate',12:'Temperate',13:'Temperate',
        14:'Temperate',15:'Temperate',16:'Temperate',17:'Cold',18:'Cold',19:'Cold',20:'Cold',
        21:'Cold',22:'Cold',23:'Cold',24:'Cold',25:'Cold',26:'Cold',27:'Cold',28:'Cold',
        29:'Polar',30:'Polar'}
    df_persist['koppen_code'] = df_persist['koppen']
    df_persist['koppen'] = df_persist['koppen_code'].map(koppen_dict)
    df_persist = df_persist.categorize(columns=['koppen']) # convert koppen column to caetory

    enc_koppen = DummyEncoder(columns=['koppen'])
    df_persist = enc_koppen.fit_transform(df_persist)
    
    # add CO2 concentration to CFE models
    if 'CFE' in model_name:
        co2 = pd.read_csv('/global/scratch/users/yanghuikang/upscale/data/site/co2_mlo_spo_filled.csv',index_col=False)
        co2 = co2.rename(columns={'average':'CO2_concentration'})
        co2['CO2_concentration'] = co2['CO2_concentration'].astype(np.float32)
        co2['year'] = co2['year'].astype(int)
        co2['month'] = co2['month'].astype(int)
        
        if 'ML' in model_name: # CFE-ML setup uses dynamic CO2 concentration
            co2_concentration = co2.loc[(co2['year']==year)&(co2['month']==month),'CO2_concentration'].values[0]
        elif 'Hybrid' in model_name: # CFE-Hybrid uses static baseline CO2 concentration
            co2_concentration = mcf.co2_2001

        df_persist['year'] = year
        df_persist['CO2_concentration'] = co2_concentration # add co2 columns
    
    print('dummy and other processing took {} seconds'.format(int(time.time()-start)), flush=True)
    
    return df_persist, ds

# reshape and convert to xarray
def reshape_2d(d_array,d_name, ds, year, month, ens_id):
    
    shape = [ds.dims['y'],ds.dims['x']]
    output_array = d_array.reshape(shape)
    x = [x.round(3) for x in ds['x']]
    y = [y.round(3) for y in ds['y']]
    output_ds = xr.DataArray(output_array,coords=[y,x],dims=['y','x'])
    
    if y[0] < 0:  # reverse y dimension if needed
        output_ds = output_ds.reindex(y=y[::-1])
    
    output_ds = output_ds.rename(d_name)
    output_ds = output_ds.to_dataset() 
    
    # scale and convert to int
    output_ds = output_ds / mcf.SCALE_FACTOR
    output_ds = output_ds.round(0).astype(np.int16)

    # add a time dimension
    ymd = str(year)+str(month).zfill(2)+'01'
    time_coord = pd.to_datetime([ymd], format="%Y%m%d")
    # ds_masked = ds_masked.to_dataset()
    output_ds = output_ds.expand_dims(dim={'time':time_coord})

    # add an ensemble dimension
    ens_coord = [ens_id]
    output_ds = output_ds.expand_dims(dim={'ensemble':ens_coord})
    
    return output_ds

def getlsm_era5():
    """
    Directly use mask from era5 data
    """
    data_path = '/global/scratch/users/yanghuikang/upscale/data/processed/utility/lsm_era5_resample.nc'
    lsm = xr.open_dataset(data_path)
    lsm['x'] = [i.round(3) for i in lsm['x']]
    lsm['y'] = [i.round(3) for i in lsm['y']]

    return lsm

# update land-sea mask
def getlsm():
    lsm_path = '/global/scratch/users/yanghuikang/upscale/data/processed/utility/lsm_1279l4_0.1x0.1.grb_v4_unpack.nc'
    lsm = xr.open_dataset(lsm_path,engine='netcdf4') # read mask

    # fix x coordiantes
    lsm = lsm.rename({'longitude':'x','latitude':'y'})
    lsm['x'] = np.append(np.arange(0,180.1,0.1),np.arange(-179.9,0,0.1))
    lsm = lsm.sortby(lsm['x'])
    lsm = lsm.squeeze('time').drop_vars(['time'])

    # resample to 0.05 resolution
    x = np.arange(-180 + 0.05 / 2., (180 + 1e-8) - 0.05 / 2., 0.05)
    y = np.arange(90 - 0.05 / 2., (-90 - 1e-8) + 0.05 / 2., -0.05)
    x = [i.round(3) for i in x]
    y = [i.round(3) for i in y]
    lsm_high = lsm.interp(coords={'x':x,'y':y},method='nearest')
    lsm_high = (lsm_high>0).astype(np.int16)

    return lsm_high

# apply veg mask
def getVegMask():
    veg_mask_file = '/global/scratch/users/yanghuikang/upscale/data/processed/utility/mcd12c1_veg_mask'
    # update vegetation mask
    veg_mask = xr.open_zarr(veg_mask_file,consolidated=False)
    veg_mask['x'] = [x.round(3) for x in veg_mask.x]
    veg_mask['y'] = [y.round(3) for y in veg_mask.y]
    veg_mask = veg_mask.astype(np.int16)

    return veg_mask

def update_mask(ds):
    """
    Post-processing the dataset:
    Apply masks: land-sea, vegetation

    """
    # cap negative predictions to zero
    ds = ds.where(ds>=0,other=0)
    
    # update mask
    # lsm_high = getlsm()
    lsm_high = getlsm_era5()
    veg_mask = getVegMask()
    ds_masked = ds.where(lsm_high['lsm']==1,other=mcf.FILL_VALUE) # np.nan is float64; the data has to be float64 to fill with nan
    ds_masked = ds_masked.where(veg_mask['veg_mask']==1,other=mcf.FILL_VALUE)
    
    # set attributes
    ds_masked.attrs['SCALE_FACTOR'] = mcf.SCALE_FACTOR
    ds_masked.attrs['FILL_VALUE'] = mcf.FILL_VALUE
    
    # chunk
    ds_masked = ds_masked.chunk({'x':500,'y':500,'time':1,'ensemble':1}) # rechunk  
    
    return ds_masked


def predict_model(model_name, year, month, ens_version, ens_id, dtest, GPP_name, ds, client):
    """
    Load model and predict

    # update: added ens_version argument 02/12/2023
    """
    
    model_type = model_name.split('_')[0] + '_' +  model_name.split('_')[1].split('-')[0] + '_' + \
        model_name.split('_')[2]
    
    model_path = '/global/home/users/yanghuikang/projects/upscale/saved_models/ensemble_'+ \
        ens_version + '/' + 'xgboost_'+model_type+'_ensemble_' + str(ens_id) + '.json'
    
    if ens_id == 30: # the for the reference model
        model_path = '/global/home/users/yanghuikang/projects/upscale/saved_models/ensemble_' + \
            ens_version + '/' + 'xgboost_'+model_type+'_reference.json'
    
    # load saved model
    model = xgb.Booster()
    model.load_model(model_path)

    # predict cfe-ml GPP
    dpred = xgb.dask.predict(client,model,dtest)

    start = time.time()
    print('predicting ...',flush=True)
    dpred = dpred.compute()
    print('predicting ml took {} seconds'.format(int(time.time()-start)))
    
    # post-process
    start = time.time()
    dpred_ds = reshape_2d(dpred, GPP_name, ds, year, month, ens_id)
    dpred_ds_masked = update_mask(dpred_ds)
    print('postprocess took {} seconds'.format(int(time.time()-start)))    
    
    return dpred_ds_masked


def get_hybrid_pred(year, month, dpred_ref_ds):
    """
    compute hybrid results based on GPP_ref
    """

    # get theoretical co2 effect for the hybrid model
    # start = time.time()
    co2_path = '/global/scratch/users/yanghuikang/upscale/data/processed/monthly/' + \
     'CFE/zarr_yearly/co2scalar/' + str(year)
    ds_co2 = xr.open_zarr(co2_path, consolidated=False)
    ds_co2 = ds_co2.sel(time=ds_co2.time.dt.month.isin([month]))
    ds_co2 = ds_co2.astype(np.float32)
    ds_co2 = ds_co2.squeeze(dim='time',drop=True)
    x = np.arange(-180 + 0.05 / 2., (180 + 1e-8) - 0.05 / 2., 0.05)
    y = np.arange(90 - 0.05 / 2., (-90 - 1e-8) + 0.05 / 2., -0.05)
    ds_co2['x'] = [i.round(3) for i in x]
    ds_co2['y'] = [i.round(3) for i in y]
    
    output_hybrid_ds = dpred_ref_ds * ds_co2['co2scalar']
    output_hybrid_ds = output_hybrid_ds.rename({'GPP_ref':'GPP_hybrid'})
    # print('hybrid, took ... {} seconds'.format(int(time.time()-start)))
    
    output_hybrid_ds = output_hybrid_ds.round(0).astype(np.int16)
    output_hybrid_ds = output_hybrid_ds.where(dpred_ref_ds['GPP_ref']!=mcf.FILL_VALUE,other=mcf.FILL_VALUE)
    
    return output_hybrid_ds


def model_deploy(model_name, ens_version, output_version, year, month, client):
    """
    master function to generate monthly GPP map from a specified model
    for all ensemble members (30 ensemble + 1 reference)

    Args:
      model_name: <temporal>_<CFE>_<partition>
      client: dask client
    """
    
    print(' '.join([model_name, output_version, str(year), str(month)]), flush=True)
    
    if 'Baseline' in model_name:
        gpp_name = 'GPP_baseline'
    elif 'ML' in model_name:
        gpp_name = 'GPP_ml'
    elif 'Hybrid' in model_name:
        gpp_name = 'GPP_ref'
    
    # path_folder = '_'.join([model_name.split('_')[i] for i in [0,2]])

    # load input data into memory
    df, ds = prepareDataframe(model_name, year, month)

    # get feature list
    model_type = '_'.join(model_name.split('_')[0:2])
    if '-' in model_type:
        model_type = model_type.split('-')[0]
    ft_list = mcf.ft_setup_dict[model_type]
    # print(ft_list)
    df = df.replace([np.inf,-np.inf],np.nan) # clean data

    # convert to xgb matrix
    start = time.time()
    dtest = xgb.dask.DaskDMatrix(client,df[ft_list],silent=True)
    # print('convert to xgb matrix took {} seconds'.format(int(time.time()-start)))

    for ens_id in range(31):
        
        print('\nensemble {}'.format(ens_id))

        output_ds = predict_model(model_name, year, month, ens_version, ens_id, dtest, gpp_name, ds, client)
        # display(output_ds)

        out_path = '/global/scratch/users/yanghuikang/upscale/data/result/predict/'+output_version+'/intermediate/'+ \
            model_name + '/zarr/'
        
        if not os.path.exists(out_path):
            # Create a new directory because it does not exist
            os.makedirs(out_path)
            print("A new directory is created: {}".format(out_path))

        zarr_out = out_path + str(year)+str(month).zfill(2) + '/' + gpp_name # final destination 
        print('saving to: ', zarr_out)

        if ens_id == 0:
            output_ds.to_zarr(zarr_out,mode='w',consolidated=True)
        else:
            output_ds.to_zarr(zarr_out,mode='a',append_dim='ensemble',consolidated=True)

        if 'Hybrid' in model_name: # compute hybrid results 
            output_hybrid_ds = get_hybrid_pred(year, month, output_ds)
            zarr_out_hybrid = out_path + str(year)+str(month).zfill(2) + '/' + 'GPP_hybrid'

            print('saving to: ', zarr_out_hybrid)
            if ens_id == 0:
                output_hybrid_ds.to_zarr(zarr_out_hybrid,mode='w',consolidated=True)
            else:
                output_hybrid_ds.to_zarr(zarr_out_hybrid,mode='a',append_dim='ensemble',consolidated=True)

