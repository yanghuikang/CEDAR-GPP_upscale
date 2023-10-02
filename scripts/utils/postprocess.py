"""
Helper functions for postprocessing GPP predictions
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import glob
import numpy as np
import xarray as xr
# import xesmf as xe

import sys
sys.path.insert(0, '../utility')
import os
import model_config as mcf

def rescaleData(ds):
    """
    scale dataset and update mask
    """
    # ds = ds/ds.attrs['SCALE_FACTOR']

    # TODO: making consistent scale factor operation
    ds = ds*ds.attrs['SCALE_FACTOR']

    return ds

def getModelName(tp, cfe):
    return tp.split('_')[0] + '_' + cfe + '_' + tp.split('_')[1]

def prepareDataAnalysis(ds, update_mask=True):
    """
    Prepare data for analysis/processing
    Update mask and fix coordiantes
    """

    ds_out = ds.copy()

    # update mask
    if update_mask:
        ds_out = ds_out.where(ds!=mcf.FILL_VALUE)

    # fix coordiantes floating point
    ds_out['x'] = [k.round(3) for k in ds_out.x.values]
    ds_out['y'] = [k.round(3) for k in ds_out.y.values]

    # ds_out.assign_attrs(ds.attrs)

    return ds_out

def prepareDataSaving(ds):
    """
    Prepare dataset for writing out
    Convert to int16 and update mask
    """
    
    ds_out = ds.copy()
    ds_out = ds_out.round(0).astype(np.int16)
    ds_out = ds_out.where(ds.notnull(),other=mcf.FILL_VALUE)
    
    # fix coordiantes floating point
    ds_out['x'] = [k.round(3) for k in ds_out.x.values]
    ds_out['y'] = [k.round(3) for k in ds_out.y.values]
    
    ds_out.assign_attrs(ds.attrs)
    
    return ds_out

def ensembleMean(ds_ensemble):
    """
    Generate ensemble mean/std, and extract the reference data
    """

    ds_mean = ds_ensemble.sel(ensemble=slice(0,30)).mean(dim='ensemble')
    ds_std = ds_ensemble.sel(ensemble=slice(0,30)).std(dim='ensemble')
    ds_ref = ds_ensemble.sel(ensemble=30)
    # ds_cv = ds_std/ds_mean

    # TODO: change 'GPP_mean' to 'GPP
    ds_all = xr.merge([ds_mean.rename({'GPP':'GPP_mean'}),
                      ds_std.rename({'GPP':'GPP_std'}),
                      ds_ref.rename({'GPP':'GPP_ref'})])

    # ds_all = uniformData(ds_all, True, True)
    ds_all = ds_all.drop_vars('ensemble')

    ds_all = ds_all.assign_attrs(ds_ensemble.attrs)

    return ds_all

def getOutPath(version, tp, cfe, ym, ensemble):
    """
    args:
        tp: temporal and partition setup
        cfe: cfe setup
        ym: year month, 'YYYYMM'
        ensemble: boolean
    """
    
    dir_path = '/global/scratch/users/yanghuikang/upscale/data/result/predict/' + version
    out_dir = dir_path + '/final/'
    model_setup = getModelName(tp, cfe)
    
    if ensemble:
        ens_type = 'ensemble'
    else:
        ens_type = 'summary'
    
    out_path = out_dir + model_setup + '/monthly/' + ens_type + '/' + ym
    
    return out_path

def saveData(ds, version, tp, cfe, ym, ensemble):
    """
    for preprocessing combine (C3/C4)
    """
    
    ds = prepareDataSaving(ds)
    out_path = getOutPath(version, tp, cfe, ym, ensemble)
    ds.to_zarr(out_path,consolidated=True,mode='w')

"""
moved to "utils.py"
"""

def getFilePath(version, stage, tmp_agg, tp, cfe, year, month, ensemble):
    """
    Get file path based on product specifications

    args:
        version:
        stage: "intermedia" or "final"
        tmp_agg: "monthly" or "annual"
        tp: <temporal>_<partition>, temporal ("ST","LT") and partition ("NT","DT")
        cfe: "Baseline","CFE-ML","CFE-Hybrid","CFE-REF"
        year: YYYY
        month: 'MM' for a specific month， 'all' to read all monthly files of the year, 'None' indicate annual GPP fiels
        ensemble: boolean
    
    return:
        a single file path if only one month is specified; 
        OR a list of filenames if month is set to 'all'
    """
    
    dir_path = '/global/scratch/users/yanghuikang/upscale/data/result/predict/' 
    model_setup = getModelName(tp, cfe)
    
    year = str(year)
    if month == 'all':
        month = '*'
    else:
        month = str(month).zfill(2)
    
    if ensemble:
        ens_type = 'ensemble'
    else:
        ens_type = 'summary'
    
    if 'annual' in tmp_agg:
        month = ''
    
    if stage == 'final':
        out_path = dir_path + version + \
            '/' + stage + '/' + model_setup + '/' + tmp_agg + '/' + ens_type + '/' + year + month
    else:
        varname_dict = {'Baseline':'GPP_baseline','CFE-ML':'GPP_ml','CFE-Hybrid':'GPP_hybrid','CFE-REF':'GPP_ref'}
        varname = varname_dict[cfe]
        if cfe == 'CFE-REF': 
            model_setup = getModelName(tp, 'CFE-Hybrid')
        out_path = dir_path + version + '/' + stage + '/' + model_setup + '/zarr/' + year + month + \
            '/' + varname
    
    if month == 'all':
        out = glob.glob(out_path)
        out.sort()
    else:
        out = out_path
    
    return out

def openFile(version, stage, tmp_agg, tp, cfe, year, month, ensemble):
    """
    Open product file with xarray
    """
    
    if month == 'all':
        file_list = getFilePath(version, stage, tmp_agg, tp, cfe, year, month, ensemble)
        ds = xr.open_mfdataset(file_list,parallel=True,engine='zarr',combine_attrs="no_conflicts")
    else:
        ds = xr.open_zarr(getFilePath(version, stage, tmp_agg, tp, cfe, year, month, ensemble),
                  consolidated=False)
    
    if stage == 'intermediate':
        old_key = list(ds.keys())[0]
        ds = ds.rename({old_key:'GPP'})

    return ds


def aggAnnual(ds, skipna=False):
    """
    Aggregate to annual GPP
    """
    
    month_length = ds.time.dt.days_in_month
    ds_annual = ds*month_length
    ds_annual = ds_annual.resample(time='1Y').sum(dim='time',skipna=skipna)

    # convert time from year end to year start
    ds_annual['time'] = [pd.to_datetime(x.values) - pd.offsets.YearBegin() for x in ds_annual['time']]
    
    return ds_annual


def computeMSC(ds, msc_start_year, msc_end_year):
    # compute MSC
    
    # aggregate to MSC, year 2001 - 2016 only
    msc_start = str(msc_start_year)+'-01-01'
    msc_end = str(msc_end_year)+'-12-31'
    ds_msc = ds.sel(time=slice(msc_start,msc_end)).groupby('time.month').mean()
    
    return ds_msc

