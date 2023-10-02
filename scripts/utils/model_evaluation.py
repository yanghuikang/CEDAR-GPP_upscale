import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.metrics import r2_score
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy.stats.mstats import theilslopes
from scipy.stats import linregress
import information_metrics as im
import dask.dataframe as dd
import model_config as mcf
import pymannkendall as mk

def getMetrics(pred, obs, prefix):
    bias = np.mean(pred - obs)
    mae = np.mean(abs(pred - obs))
    rmse = np.sqrt(((pred - obs) ** 2).mean())
    mape = np.mean(abs(pred - obs) / obs) * 100
    r2score = r2_score(obs, pred)
    r = np.corrcoef(pred, obs)[0,1]
    r2 = r**2
    # slope,_,_,_,_ = linregress(obs, pred)
    
    mean = np.mean(obs)
    nRmse = rmse/mean
    
    n = len(obs)
    precision = np.sqrt(np.sum((pred-obs-bias) ** 2)/n)
    
    # values = [n, rmse, bias, precision, mae, mape, nRmse, r2, r, r2score,slope]
    values = [n, rmse, bias, precision, mae, mape, nRmse, r2, r, r2score]
    # names = ['count','rmse','bias','precision','mae','mape','nRmse','r2','r','r2score','slope']
    names = ['count','rmse','bias','precision','mae','mape','nRmse','r2','r','r2score']
    
    if prefix != None:
        names = [prefix+'_'+x for x in names]
    
    out = pd.Series(dict(zip(names, values)))    
    
    return out

def applyMetrics(data, pred, obs):
    data_copy = data.dropna(subset=[pred,obs])
    return getMetrics(data_copy[pred],data_copy[obs],None)

def renameModel(data):
    """
    Rename models for CV results
    Create new columns indicating type of GPP, temporal range, CFE, and model setup
    """
    data_out = data.rename(columns={'model_name':'model_name_full'})
    data_out['GPP_type'] = data_out['model_name_full'].str.split('_',expand=True)[2]
    data_out['Temporal_type'] = data_out['model_name_full'].str.split('_',expand=True)[0]
    data_out['CFE_type'] = data_out['model_name_full'].str.split('_',expand=True)[1]
    data_out['Model'] = data_out['model_name_full'].str.split('_').str[0:2].str.join('_')

    data_out = data_out.dropna(subset=['_pred'])

    # model_list = ['ST_Baseline','ST_CFE-ML','ST_CFE-Hybrid','LT_Baseline','LT_CFE-ML','LT_CFE-Hybrid']
    # model_cat = pd.CategoricalDtype(model_list,ordered=True)

    cfe_list = ['Baseline','CFE-ML','CFE-Hybrid']
    cfe_cat = pd.CategoricalDtype(cfe_list,ordered=True)

    if 'group' in data.columns:
        group_list = ['Monthly','MSC','Anomaly','Trend','Cross-site']
        group_dict = {'Overall':'Monthly','IAV-month':'Anomaly'}
        group_cat = pd.CategoricalDtype(group_list)
        data_out['group'] = data_out['group'].replace(group_dict)

    # data_out = data_out[data_out['Model'].isin(model_list)]
    # data_full['Model'] = data_full['Model'].astype(model_cat)
    cfe_list = ['Baseline','CFE-ML','CFE-Hybrid']
    cfe_cat = pd.CategoricalDtype(cfe_list,ordered=True)

    # data_full = data_full[data_full['group'].isin(group_list)]
    # data_full_plot['group'] = data_full_plot['group'].astype(group_cat)
    data_out['CFE_type'] = data_out['CFE_type'].astype(cfe_cat)

    # group IGBP types
    data_out['PFT'] = data_out['IGBP'].replace(mcf.pft_dict)
    
    return data_out


# MSC and IAV data
def getIAV(data, group_vars, obs_col, pred_col, year_col, min_year, trend_method='ols'):
    
    msc_group = group_vars + ['month']
    iav_group = group_vars + [year_col,'month']
    
    # MSC
    data_msc = data.groupby(msc_group)[[obs_col,pred_col]].mean().reset_index()
    
    data_msc_count = data.groupby(msc_group)[obs_col].count().reset_index()
    data_msc_count = data_msc_count.rename(columns={obs_col:'count'})
    data_msc = data_msc.merge(data_msc_count)
    # data_msc = data_msc[data_msc['count']>1]
    
    # IAV
    data_iav = data.groupby(msc_group)[[obs_col,pred_col]].transform(lambda x: x - x.mean())
    data_iav = pd.concat([data[iav_group],data_iav],axis=1)
    data_iav['count'] = data.groupby(msc_group)[obs_col].transform(lambda x: x.count())
    # data_iav = data_iav[data_iav['count']>1]
    
    # IAV standardized
    data_iav_sd = data_iav.groupby(msc_group)[[obs_col,pred_col]] \
        .apply(lambda x: x.div(x.max()-x.min()))
    # data_iav_sd = pd.concat([data_iav[iav_group],data_iav_sd],axis=1)
    # data_iav_sd = data_iav_sd.replace([np.inf,-np.inf],np.nan)
    # data_iav_sd = data_iav_sd.dropna()
    
    # across-site
    data_site = data_msc.groupby(group_vars)[[obs_col,pred_col]].mean().reset_index()
    
    data_msc['group'] = 'MSC'
    data_iav['group'] = 'IAV-month'
    data_overall = data[iav_group+[obs_col,pred_col]]
    data_overall['group'] = 'Overall'
    data_site['group'] = 'Cross-site'
    
    # IAV-annual
    data_annual = getAnnualGPP(data, [obs_col,pred_col], year_col, group_vars, screen_site=True, min_year=min_year)
    data_annual['group'] = 'IAV-annual'
    
    # site trend annual
    data_trend_annual = data_annual.groupby(group_vars).apply(lambda x: getTrend(x, year_col, obs_col, pred_col, trend_method)).reset_index()
    data_trend_annual['group'] = 'Trend_annual'

    # site trend monthly
    data_cp = data.copy()
    data_cp = data_cp.dropna(subset=[obs_col, pred_col])
    
    # test removing sites with records less than 1 year
    site_month_counts = data_cp.groupby(['SITE_ID'],as_index=False).agg({'month':'count'}).reset_index()
    site_month_counts = site_month_counts[site_month_counts['month']>12]
    data_cp = data_cp[data_cp['SITE_ID'].isin(site_month_counts['SITE_ID'].unique())]
    
    data_cp['datetime'] = pd.to_datetime(dict(year=data_cp[year_col],month=data_cp['month'],day=1))
    data_cp['num_month'] = ((data_cp['datetime']-pd.to_datetime('1999-12-01'))/np.timedelta64(1,'M')).values.round()
    data_trend_monthly = data_cp.groupby(group_vars).apply(lambda x: getTrend(x,'num_month',obs_col,pred_col,trend_method)).reset_index()
    data_trend_monthly['group'] = 'Trend'
    
    out = data_overall.append(data_msc).append(data_iav).append(data_annual).append(data_trend_annual) \
        .append(data_trend_monthly).append(data_site)
    
    return out


def getTrend(data, x_col, y_obs, y_pred, method='ols'):
    """
    Compute trend on annual year for observed and predicted GPP
    """
    # print(data.shape[0])

    if method == 'ols':
        X = data[x_col]
        X = X - data[x_col].min()
        X = sm.add_constant(X)

        # get observed trend
        ols_model_obs = sm.OLS(data[y_obs], X)
        ols_obs_result = ols_model_obs.fit()
        obs_trend = ols_obs_result.params[x_col]
        obs_pvalue = ols_obs_result.pvalues[1]

        # ols_model_obs = ols(y_obs+' ~' + x_col, data)
        # ols_result_obs = ols_model_obs.fit()

        # get predicted trend
        ols_model_pred = sm.OLS(data[y_pred], X)
        ols_pred_result = ols_model_pred.fit()
        pred_trend = ols_pred_result.params[x_col]
        pred_pvalue = ols_pred_result.pvalues[1]

        # ols_model_pred = ols(y_pred+' ~' + x_col, data)
        # ols_result_pred = ols_model_pred.fit()

        # print(ols_result_obs.params)

        # obs_trend = ols_result_obs.params[x_col]
        # pred_trend = ols_result_pred.params[x_col]

    else: 
        mk_model_obs = mk.original_test(data[y_obs])
        sen_slope_obs = mk_model_obs.slope
        # sen_intercept = mk_model['intercept']
        sen_pvalue_obs = mk_model_obs.p
        # sen_slope_obs,_,lo_slope,up_slope = theilslopes(data[y_obs],data[x_col])
        # sen_slope_pred,_,_,_ = theilslopes(data[y_pred],data[x_col])

        mk_model_pred = mk.original_test(data[y_pred])
        sen_slope_pred = mk_model_pred.slope
        # sen_intercept = mk_model['intercept']
        sen_pvalue_pred = mk_model_pred.p

        obs_trend = sen_slope_obs
        pred_trend = sen_slope_pred
        obs_pvalue = sen_pvalue_obs
        pred_pvalue = sen_pvalue_pred

    out = pd.Series({y_obs:obs_trend,
                    y_pred:pred_trend,
                    y_obs+'_pvalue':obs_pvalue,
                    y_pred+'_pvalue':pred_pvalue,
                    'count':data.shape[0]})
    
    return out


def getTrend_single(data, x_col, y_col):
    
    X = data[[x_col]]
    X = X - data[x_col].min()
    X = sm.add_constant(X)
    y = data[y_col]

    ols_model = sm.OLS(y, X)
    ols_result = ols_model.fit()
    ols_slope = ols_result.params[x_col]
    ols_intercept = ols_result.params['const']
    ols_pvalue = ols_result.pvalues[1]
    # print(list(ols_result))
    
    # sen_slope,sen_intercept,lo_slope,up_slope = theilslopes(data[y_col],data[x_col]-data[x_col].min())
    mk_model = mk.original_test(data[y_col])
    sen_slope = mk_model.slope
    sen_intercept = mk_model.intercept
    sen_pvalue = mk_model.p
    
    return {'ols_slope':ols_slope, 'sen_slope':sen_slope, 'ols_intercept':ols_intercept,'sen_intercept':sen_intercept,
        'ols_pvalue': ols_pvalue, 'sen_pavlue':sen_pvalue}


def getTrend_GPPanomaly(data, group_vars, gpp_col, year_col='year', screen_site=True, min_year=5):
    """
    get trend from aggregated GPP anomaly
    
    """
    data1 = data.copy()
    
    # site vars
    site_vars = ['SITE_ID'] # do not use MODIS PFT since it may be dynamic for ST model
    site_vars = [x for x in site_vars if x in data.columns]
    
    # get annual gpp
    annual = getAnnualGPP(data1, [gpp_col], year_col, group_vars, screen_site=screen_site, min_year=min_year)
    annual = annual.rename(columns={gpp_col:'GPP'})
    
    # get GPP anomaly
    annual['GPP_anomaly'] = annual.groupby(group_vars+site_vars)['GPP'].transform(lambda x: x - x.mean())
    annual_long = annual.melt(id_vars=site_vars+group_vars+[year_col]+['count'],value_vars=['GPP','GPP_anomaly'],var_name='GPP_type',
                              value_name='GPP')
    # annual_agg = None
    annual_agg = annual_long.groupby(group_vars+[year_col]+['GPP_type'])['GPP'].agg(['mean','count']).reset_index()
    annual_agg = annual_agg.rename(columns={'mean':'GPP'})
    
    # get trend/slope
    def computeTrend(data, x_col, y_col):
        return pd.Series(getTrend_single(data, x_col, y_col))
    
    annual_trend = annual_agg.groupby(['GPP_type']+group_vars).apply(lambda x: computeTrend(x, year_col, 'GPP')).reset_index()
    # annual_trend = None
    
    return annual_trend, annual_agg, annual_long


def getAnnualGPP(data, gpp_cols, year_col, group_vars, screen_site=False, min_year=5):
    """
    Compute annual GPP from monthly data (weigthed by days in month)
    
    min_year: minumum number of years
    
    Unit: /year

    """
    data_new = data.copy()
    
    site_vars = ['SITE_ID','koppen','IGBP'] # do not use MODIS PFT since it may be dynamic for ST model
    site_vars = [x for x in site_vars if x in data.columns]
    site_vars = [x for x in site_vars if x not in group_vars]
    index_cols = site_vars+ group_vars + [year_col]
    
    # compute annual GPP
    data_new['day'] = 1 # dummy day column in order to convert to datetime
    if isinstance(data_new, dd.DataFrame):
        data_new['dtime'] = dd.to_datetime(data_new[['year','month','day']])
    else:
        data_new['dtime'] = pd.to_datetime(dict(year=data_new[year_col],month=data_new['month'],day=data_new['day']))
    
    ####
    # count number of months per year before interpolation
    annual_count = data_new.groupby(index_cols)[gpp_cols[0]].count()
    annual_count = annual_count.rename('count')
    annual_count = annual_count.reset_index()
    
    def fill_in_and_interpolate(df):
        start_date = df['dtime'].min()
        end_date = df['dtime'].max()
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

        df_complete = pd.DataFrame({'dtime': date_range})

        # merge the two dataframes to fill in the missing months
        df_complete = pd.merge(df_complete, df, on='dtime', how='left')

        # interpolate missing values
        for col in gpp_cols:
            df_complete[col] = df_complete[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

        return df_complete
    
    data_complete = data_new.groupby(site_vars+group_vars).apply(fill_in_and_interpolate).reset_index(drop=True)
    # print(data_complete.shape,data_new.shape)
    data_new = data_complete
    ####

    data_new['month_length'] = data_new['dtime'].dt.days_in_month
    gpp_month_cols = [x+'_month' for x in gpp_cols]

    for col in gpp_cols: # monthly GPP sums
        gpp_month_col = col + '_month'
        data_new[gpp_month_col] = data_new[col]*data_new['month_length']

    #### keep site-year with a minimum of 11 months data
    # annual_count = data_new.groupby(index_cols)[gpp_cols[0]].count()
    # annual_count = annual_count.rename('count')
    # annual_count = annual_count.reset_index()
    ####

    annual = data_new.groupby(index_cols)[gpp_month_cols].sum().reset_index()
    # annual_count = data_new.groupby(index_cols)[gpp_cols[0]].count()
    # annual_count = annual_count.rename('count')
    # annual_count = annual_count.reset_index()
    annual = annual.merge(annual_count,on=index_cols)
    annual = annual.rename(columns=dict(zip(gpp_month_cols,gpp_cols)))
    
    # a minimum of 11 months to compute annual
    annual_clean = annual[annual['count']>10]
    
    if screen_site:
        g_cols = group_vars
        if 'SITE_ID' not in group_vars:
            g_cols = g_cols + ['SITE_ID']
        # site_annual = annual_clean.groupby(group_vars+['SITE_ID']).agg(count_year=pd.NamedAgg(gpp_cols[0],'count')).reset_index()
        site_annual = annual_clean.groupby(g_cols)[gpp_cols[0]].agg('count').reset_index()
        site_list = site_annual[site_annual[gpp_cols[0]]>=min_year]['SITE_ID'].unique()
        # site with more than 5 years of data
        annual_clean = annual_clean[annual_clean['SITE_ID'].isin(site_list)]
    
    return annual_clean


def getDataLimits(data, group_vars, x_var, y_var, max_perc=1):
    """
    Get the limits of combined x and y variable of data
    Create a dummy data limits dataset to set equal x and y range for facet plotting
    """
    data_new = data.copy()
    
    if len(group_vars) == 0:
        data_new['group'] = 1 # create a dummy group col
        group_vars = ['group']

    data_min = data_new.groupby(group_vars)[[x_var, y_var]].min().min(axis=1).reset_index()
    
    if max_perc == 1:
        data_max = data_new.groupby(group_vars)[[x_var, y_var]].max().max(axis=1).reset_index()
    else:
        data_max = data_new.groupby(group_vars)[[x_var, y_var]].quantile(max_perc).max(axis=1).reset_index()
    data_limits = data_min.append(data_max)
    data_limits = data_limits.rename(columns={0: x_var})
    data_limits[y_var] = data_limits[x_var]
    
    return data_limits


def getErrorText(data,group_vars,x_var,y_var,add_slope=False,position='top',y_scale=8,x_scale=20,max_perc=1):
    """
    Genereate goodness-of-fit metrics text for scatter plots
    """

    y_axis_buffer = y_scale
    x_axis_buffer = x_scale

    if add_slope:
        y_axis_buffer = 4
    
    error_group = data.groupby(group_vars).apply(lambda x: applyMetrics(x,y_var,x_var))
    
    if max_perc == 1:
        error_group['max'] = data.groupby(group_vars)[[x_var,y_var]].max().max(axis=1)
    else:
        error_group['max'] = data.groupby(group_vars)[[x_var,y_var]].quantile(0.997).max(axis=1)

    error_group['min'] = data.groupby(group_vars)[[x_var,y_var]].min().min(axis=1)
    
    # determine location of the text
    if position == 'bottom': # lower right
        error_group['x'] = error_group['max'] - (error_group['max']-error_group['min'])/x_axis_buffer
        error_group['y'] = error_group['min'] + (error_group['max']-error_group['min'])/y_axis_buffer
    else:   # upper left
        error_group['x'] = error_group['min'] + (error_group['max']-error_group['min'])/x_axis_buffer
        error_group['y'] = error_group['max'] - (error_group['max']-error_group['min'])/y_axis_buffer
    
    error_group['RMSE_label'] = 'RMSE: '+ error_group['rmse'].round(2).apply(str)
    error_group['R2_label'] = '$R^{2}$: '+error_group['r2score'].round(2).apply(str)
    
    if 'slope' in error_group.columns:
        error_group['slope'] = 'slope: '+error_group['slope'].round(2).apply(str)

    error_group['label'] = error_group['RMSE_label'] + '\n' + error_group['R2_label']

    if add_slope:
        error_group['label'] = error_group['label'] + '\n' + error_group['slope']

    text_plot = error_group.reset_index()
    return text_plot

print('load model_evaluation')


from scipy.interpolate import interpn
def get_density(data,x_col,y_col,sort=True,bins = [50,50]):
    """
    Get density estimates 
    """
    
    x = np.array(data[x_col])
    y = np.array(data[y_col])
    
    data0 , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data0 , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    
    # log scale
    z_log = np.log(z+1)

    data_density =  pd.DataFrame({x_col:x, y_col:y, 'z':z, 'z_log':z_log})
    
    return data_density
