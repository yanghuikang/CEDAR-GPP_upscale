#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Postprocessing:

Combine model predictions in proportion to C3/C4 percentage on monthly data
Generate ensemble summary dataset

Run on savio

Update:
Make it work for long-term models which don't have CFE-ML settings

@author: yanghuikang
"""

import warnings
warnings.filterwarnings('ignore')

import time
import math
import xarray as xr
import sys
sys.path.insert(0, '/global/home/users/yanghuikang/projects/upscale/scripts/utility')
import postprocess as ptp
import model_config as mcf
import pandas as pd

# set temperary directory for data spill to scratch, otherwise by default data will spill to the home directory
import dask
dask.config.set({'temporary_directory': '/global/scratch/users/yanghuikang/upscale/data/temp'})
from dask.distributed import Client, LocalCluster


def combineModelPredictions(version, tp_setup, ym):
    """
    Args:
        version: model version
        tp_setup: <temporal>_<partition>
        ym: year month 'YYYYMM'
    """

    temporal = tp_setup.split('_')[0]
    partition = tp_setup.split('_')[1]

    # get data paths
    dir_path = '/global/scratch/users/yanghuikang/upscale/data/result/predict/' + version
    data_baseline_path = dir_path + '/intermediate/' + ptp.getModelName(tp_setup,'Baseline') + '/zarr/' + ym + '/GPP_baseline'
    data_ml_path = dir_path + '/intermediate/' + ptp.getModelName(tp_setup,'CFE-ML') + '/zarr/' + ym + '/GPP_ml'
    data_hybrid_path = dir_path + '/intermediate/' + ptp.getModelName(tp_setup,'CFE-Hybrid') + '/zarr/' + ym + '/'
    
    # read GPP prediction data
    ds_baseline = xr.open_zarr(data_baseline_path) # baseline
    if temporal == 'ST':
        ds_ml = xr.open_zarr(data_ml_path) # CFE-ML

    file_list_hybrid = [data_hybrid_path + 'GPP_ref', data_hybrid_path + 'GPP_hybrid'] # Hybrid
    ds_hybrid = xr.open_mfdataset(file_list_hybrid,parallel=True,engine='zarr',combine_attrs="no_conflicts")
    
    # open C3/C4 dataset
    c4_path = '/global/scratch/users/yanghuikang/upscale/data/processed/monthly/C4_005/zarr_yearly/C4'
    ds_c4 = xr.open_zarr(c4_path, consolidated=False, chunks={'x':500,'y':500})
    ds_c4 = ds_c4/100 # scale

    # unify data formats
    ds_baseline = ptp.prepareDataAnalysis(ds_baseline)
    if temporal == 'ST':
        ds_ml = ptp.prepareDataAnalysis(ds_ml)
    ds_hybrid = ptp.prepareDataAnalysis(ds_hybrid)
    ds_c4 = ptp.prepareDataAnalysis(ds_c4,False)

    # re-scale GPP_ref for long-term towards 1982 baseline
    if temporal == 'LT':
        zarr_path = '/global/scratch/users/yanghuikang/upscale/data/processed/monthly/' 
        co2_path = zarr_path + 'CFE/zarr_yearly/co2scalar_1982' + '/' + ym[0:4]
        co2_scalar_1982 = xr.open_zarr(co2_path, consolidated=False, chunks={'x':500,'y':500})

        # fix x, y coordiantes
        co2_scalar_1982['x'] = [k.round(3) for k in co2_scalar_1982.x.values]
        co2_scalar_1982['y'] = [k.round(3) for k in co2_scalar_1982.y.values]

        # fix time coordiante
        co2_time = co2_scalar_1982['time'].to_index()
        co2_time = co2_time - pd.offsets.MonthBegin(1)
        co2_scalar_1982['time'] = co2_time

        ds_hybrid['GPP_ref'] = ds_hybrid['GPP_ref'] * co2_scalar_1982['co2scalar']
        # print(co2_scalar_1982)
        # print(ds_hybrid['GPP_ref'])
    
    # apply CFE in proportion to C4 percentage
    if temporal == 'ST':
        ds_ml_final = ds_ml['GPP_ml'] * (1 - ds_c4['C4']) + ds_hybrid['GPP_ref'] * ds_c4['C4']
    ds_hybrid_final = ds_hybrid['GPP_hybrid'] * (1 - ds_c4['C4']) + ds_hybrid['GPP_ref'] * ds_c4['C4']

    # convert, rename, and change data type
    if temporal == 'ST':
        ds_ml_final = ds_ml_final.to_dataset(name='GPP')
    ds_hybrid_final = ds_hybrid_final.to_dataset(name='GPP')
    ds_baseline = ds_baseline.rename({'GPP_baseline':'GPP'})
    ds_ref = ds_hybrid[['GPP_ref']].rename({'GPP_ref':'GPP'})
    
    # udpate attributes
    def updateAttrs(ds, temporal, partition, cfe, fill_value, scale_factor):

        ds_out = ds.copy()

        ds_out.attrs['FILL_VALUE'] = fill_value
        ds_out.attrs['SCALE_FACTOR'] = scale_factor
        ds_out.attrs['Temporal_Setup'] = temporal
        ds_out.attrs['CFE_Setup'] = cfe
        ds_out.attrs['Partition_Setup'] = partition

        return ds_out

    # TODO: fix inconsistent scale factors (to 0.01 rather than 100)
    attrs = {'temporal':temporal,'partition':partition,'fill_value':mcf.FILL_VALUE,'scale_factor':mcf.SCALE_FACTOR}
    attrs.update({'cfe':'Baseline'})
    ds_baseline = updateAttrs(ds_baseline, **attrs)
    if temporal == 'ST':
        attrs.update({'cfe':'CFE-ML'})
        ds_ml_final = updateAttrs(ds_ml_final, **attrs)
    attrs.update({'cfe':'CFE-Hybrid'})
    ds_hybrid_final = updateAttrs(ds_hybrid_final, **attrs)
    attrs.update({'cfe':'CFE-REF'})
    ds_ref = updateAttrs(ds_ref, **attrs)
    
    # compute ensemble summaries
    ds_baseline_summary = ptp.ensembleMean(ds_baseline)
    if temporal == 'ST':
        ds_ml_final_summary = ptp.ensembleMean(ds_ml_final)
    ds_hybrid_final_summary = ptp.ensembleMean(ds_hybrid_final)
    ds_ref_summary = ptp.ensembleMean(ds_ref)
    
    # save data
    ptp.saveData(ds_baseline, version, tp_setup, 'Baseline', ym, True)
    if temporal == 'ST':
        ptp.saveData(ds_ml_final, version, tp_setup, 'CFE-ML', ym, True)
    ptp.saveData(ds_hybrid_final, version, tp_setup, 'CFE-Hybrid', ym, True)
    ptp.saveData(ds_ref, version, tp_setup, 'CFE-REF', ym, True)
    ptp.saveData(ds_baseline_summary, version, tp_setup, 'Baseline', ym, False)
    if temporal == 'ST':
        ptp.saveData(ds_ml_final_summary, version, tp_setup, 'CFE-ML', ym, False)
    ptp.saveData(ds_hybrid_final_summary, version, tp_setup, 'CFE-Hybrid', ym, False)
    ptp.saveData(ds_ref_summary, version, tp_setup, 'CFE-REF', ym, False)

    
def main(argv):
    """
    Argvs:
        version: version
        tp_setup: <temporal>_<partition>
        year: year
    """
    
    # Dask local cluster
    cluster = LocalCluster(ip='*') 
    client = Client(cluster)
    print(client.dashboard_link, flush=True)
    
    version = argv[0] # 'v1_2_0'
    tp_setup = argv[1] # 'ST_NT'
    year = argv[2] # '2001'
    print(version, year, tp_setup, flush=True)
    
    start = time.time()

    for month in range(12):

        month = month+1
        ym = year + str(month).zfill(2)

        print(year, month,flush=True)
        combineModelPredictions(version, tp_setup, ym)
    
    print('took {} minutes'.format(round((time.time()-start)/60),2))

    
if __name__ == '__main__':
    main(sys.argv[1:])