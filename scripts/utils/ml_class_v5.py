import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

import pandas as pd
from pandas.api.types import CategoricalDtype
import time
import datetime as dt
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import numpy as np

from dask_ml.wrappers import ParallelPostFit
import dask.dataframe as dd
import dask.array as da

import model_config as mcf



def getMetrics(pred, obs, prefix):
    bias = np.mean(pred - obs)
    mae = np.mean(abs(pred - obs))
    rmse = np.sqrt(((pred - obs) ** 2).mean())
    mape = np.mean(abs(pred - obs) / obs) * 100
    r2score = r2_score(obs, pred)
    r = np.corrcoef(pred, obs)[0,1]
    r2 = r**2
    
    mean = np.mean(obs)
    nRmse = rmse/mean
    
    n = len(obs)
    precision = np.sqrt(np.sum((pred-obs-bias) ** 2)/n)
    
    values = [n, rmse, bias, precision, mae, mape, nRmse, r2, r, r2score]
    names = ['count','rmse','bias','precision','mae','mape','nRmse','r2','r','r2score']
    
    if prefix != None:
        names = [prefix+'_'+x for x in names]
    
    out = pd.Series(dict(zip(names, values)))    
    
    return out

def applyMetrics(data, pred, obs):
    data = data.dropna(subset=[pred,obs])
    return getMetrics(data[pred],data[obs],None)

class Model:
    """
    Basic class to run a ML experiment with cross-validation
    exp: information for running the experiment
    """
    
    def __init__(self,data,config):

        """
        Options:
        <Model specifications>
        target: column name of the target variable for prediction
        features: list of column names used as training features
        model: name of the machine learning model; options: 'lasso','svr','rf','nn','xgboost'
        group: column name of the variable that groups the training samples (usually FLUX SITE ID)
        hybrid_model: boolean; add biophysical CO2 sensitivity to the model predictions;
                      only use when if CO2_concentration is a feature,
                      will produce both "CFE_ML" and "CFE_hybrid" predictions;

        <Training specifications>
        hyper_tune: boolean; whether to perform hyper paramter tuning in training
        cross_validation: boolean; whether or not to perform 5-fold stratified group cross validation
        ml_params: dictionary; specify model hyperparameters to use, if hyperparameter tuning not performed

        <Other specifications>
        RANDOM_STATE: random state number for deterministic analysis
        NUM_CORES: number of cpu cores avaialabe for parallelization

        """
        
        self.data=data
        self.config = config
        self.rf_param = 'big'
        self.num_jobs = 20
        self.cross_validation = True
        self.clf = None
        self.parallel_predict = False
        
        # hybrid model indicator, if true, prediction will include hybrid model results and hybrid-ref results
        self.hybrid_model = False  
        
        # id variables
        id_vars = ['SITE_ID','year','month','IGBP','MODIS_PFT','koppen']
        self.id_vars = [x for x in id_vars if x in data.columns]   

        if "RANDOM_STATE" in config:
            self.RANDOM_STATE = config['RANDOM_STATE']
        else:
            self.RANDOM_STATE = mcf.RANDOM_STATE
            
        if "rf_param" in config:
        	self.rf_param = config['rf_param']
            
        if "NUM_CORES" in config:
            self.NUM_CORES = config['NUM_CORES']
            self.num_jobs = self.NUM_CORES - 4
            
        if "hyper_tune" in config:
            self.hyper_tune = config['hyper_tune']
        else:
            self.hyper_tune = True
        
        if "ml_params" in config:
            self.ml_params = config['ml_params']
        else:
            self.ml_params = None
        
        if 'cross_validation' in config:
            self.cross_validation = config['cross_validation']
        
        if 'hybrid_model' in config:
            self.hybrid_model = config['hybrid_model']            
            
        if self.hyper_tune == False:
            # hyperparameter settings, if not provided, will use default
            self.regressor_n_thread = self.num_jobs
        else:
            self.regressor_n_thread = 1
            
        if 'parallel_predict' in config:
            self.parallel_predict=config['parallel_predict']
        else:
            self.parallel_predict = False
        
        if self.hyper_tune:
            self.parallel_predict = False
            print('Warning: dask parallel predict does not work with hyper tuning for now...')
        
    def trainML(self,train, test, X_train, X_test, y_train, y_test, model, group, target, id_vars):
            
        if model == 'rf':
            
            max_features = min(15, X_train.shape[1])
            
            default_args = {'max_features':max_features, 'min_samples_leaf': 1, 'min_samples_split':4,
                            'n_estimators':200, 'n_jobs': self.regressor_n_thread, 
                            'random_state':self.RANDOM_STATE}
            if self.ml_params is not None: # if parameters are provided, update them
                model_args = default_args.copy()
                model_args.update(self.ml_params)
            else:
                model_args = default_args          
            
            clf = RandomForestRegressor(**model_args)        
            params_big = {
                'n_estimators':[100,300,500],
                'max_depth':[5,15,None],
                'min_samples_leaf':[5,10],
                'min_samples_split':[8,12,16]
                }
            params_small = {
                'n_estimators':[20,30,50],
                'max_depth':[5,10],
                'min_samples_leaf':[5,15],
                'min_samples_split':[8,12]            
            }
            if self.rf_param == 'small':
            	params = params_small
            else:
            	params = params_big
                
        elif model == 'svr':
            default_args = {'kernel':'rbf','gamma':'auto'}

            if self.ml_params is not None: # if parameters are provided, update them
                model_args = default_args.copy()
                model_args.update(self.ml_params)
            else:
                model_args = default_args
            
            clf = Pipeline(
                steps = [
                    ('scaler',StandardScaler()),
                    ('svr',
                     SVR(**model_args))])
            params = {
                'svr__C':[0.01, 0.1,1,10,100,1000],
                'svr__gamma':[0.1,1,10,'auto','scale']
            }

        elif model == 'lasso':
            default_args = {'max_iter': 1e6, 'alpha':1e-1}

            if self.ml_params is not None: # if parameters are provided, update them
                model_args = default_args.copy()
                model_args.update(self.ml_params)
            else:
                model_args = default_args
                
            clf = Pipeline(
                steps = [
                    ('scaler',StandardScaler()),
                    ('lasso',Lasso(**model_args))])
            params = {
                'lasso__alpha':[1e-4,1e-3,1e-2,1e-1]
            }
            # clf = LassoCV(max_iter = 1000000, cv = train_cv, n_jobs=self.num_jobs)
            
        elif model == 'xgboost':
            
            default_args = {'colsample_bytree':0.4603, 'gamma':0, 'learning_rate':0.05, 'max_depth':6,
                           'min_child_weight':7, 'n_estimators': 100, 'min_child_weight':7,
                           'reg_alpha':0.5, 'reg_lambda': 0.8, 'subsample': 0.5, 'verbosity': 0,
                           'random_state':self.RANDOM_STATE, 'nthread':self.regressor_n_thread}
            
            if self.ml_params is not None: # if parameters are provided, update them
                model_args = default_args.copy()
                model_args.update(self.ml_params)
            else:
                model_args = default_args
                
            clf = xgb.XGBRegressor(**model_args)
            # print(default_args,clf)
            
            params = {
                    'learning_rate':[0.001,0.01,0.05],
                    'gamma':[0,0.01,0.1],
                    'colsample_bytree':[0.1,0.2,0.3],
                    'n_estimators':[100,300,500]
                     }
            
        elif model == 'nn':
            
            default_args = {'activation':'relu', 'max_iter':1000, 'early_stopping':True, 
               'random_state':self.RANDOM_STATE}
            
            if self.ml_params is not None: # if parameters are provided, update them
                model_args = default_args.copy()
                model_args.update(self.ml_params)
            else:
                model_args = default_args
            
            clf = Pipeline(
                steps = [
                    ('scaler',StandardScaler()),
                    ('mlp',
                     MLPRegressor(**model_args))])
            params = {
                'mlp__hidden_layer_sizes':[(32,32,32),(64,64,64),(128,128,128),
                                           (32,32,32,32,32),(64,64,64,64,64),(128,128,128,128,128)],
                'mlp__learning_rate_init':[0.001,0.0001,0.00001],
                'mlp__batch_size':[64,128]
            }
        
        # timer up
        start_time = time.time()   
        
        if self.hyper_tune:
            train_cv = GroupKFold(n_splits=5) # randomized hyperparameter search
            clf = RandomizedSearchCV(clf, param_distributions=params, n_iter=25,
                                     scoring='neg_mean_squared_error', n_jobs=self.num_jobs,
                                     cv = train_cv, random_state=self.RANDOM_STATE)
            clf.fit(X_train, y_train, groups=train[group])
            best_params = clf.best_params_   
        else:
            if self.parallel_predict:
                clf = ParallelPostFit(estimator=clf)  # dask-ml parallelize prediction
            clf.fit(X_train, y_train)
            
        search_time = round(time.time() - start_time, 1)
        print('--- Fitting: %s seconds ----' % (search_time))
        
        # predict for trianing
        train_pred = clf.predict(X_train)
        
        # predict for testing
        test_pred = clf.predict(X_test)
        pred = test[id_vars].copy()
        pred.loc[:,target] = y_test
        pred.loc[:,'_pred'] = test_pred
        
        # create hybrid predictions
        if self.hybrid_model:
            pred_hybrid = self.predict_hybrid(clf=clf, data=test)
            pred_hybrid.index = pred.index # need to set index
            pred = pd.concat([pred,pred_hybrid],axis=1)
            # for col in pred_hybrid.columns:
            #    pred.loc[:,col] = pred_hybrid[col]
            # pred.loc[:,target+'_pred_hybrid_ref'] = self.predict(clf=clf, data=test, fix_CO2=True)
            # pred['CO2_effect'] = getCO2effect_simple((test['Tmean']-273.15),test['CO2_concentration'])
            # pred.loc[:,target+'_pred_hybrid'] = pred[target+'_pred_hybrid_ref'] * pred['CO2_effect']
        
        # get error metrics
        error_train = getMetrics(train_pred, y_train, 'train')
        error_test = getMetrics(test_pred, y_test, 'test')
        
        # save trian and test errors
        row = pd.Series({'target':target, 'model':model})
        row = row.append(error_train)
        row = row.append(error_test)
    
        if self.hyper_tune==True:
            row = row.append(pd.Series(best_params))
            
        return row, pred, clf
    
    def updateModelArgs(args):
        for key, value in self.ml_params.items():
            args[key] = value
        return args
    
    def runML(self):
        
        features = self.config['features']
        target = self.config['target']
        group = self.config['group']
        model = self.config['model']
        name = self.config['name']
        data = self.data
        
        if model == 'xgboost':
            data = data.dropna(subset=features,how='all')
            # print(data.shape)
        else:
            data = data.dropna(subset=features)
        
        X = np.array(data[features])
        y = np.array(data[target])
        
        id_vars = self.id_vars
        
        if self.cross_validation:
            # split the data, grouped by SITE_ID, stratified by IGBP
            n_fold = 5
            stratify_class = data['MODIS_IGBP'] 
            group_kfold = StratifiedGroupKFold(n_splits=n_fold,shuffle=True,random_state=self.RANDOM_STATE)
            group_kfold.get_n_splits(X,stratify_class,data[group])

            # prepare data containers
            errors = pd.DataFrame()
            predictions = pd.DataFrame()
            best_params_all = pd.DataFrame()

            i = 0 # initiate fold number
            for train_index, test_index in group_kfold.split(X, stratify_class, data[group]):

                print('fold:', i, flush=True)
                train = data.iloc[train_index,:]
                test = data.iloc[test_index,:]
                # print(train.shape, test.shape)
                # print(train.groupby(['SITE_ID'])['GPP'].count())
                # print(test.groupby(['SITE_ID'])['GPP'].count())
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                row, pred, clf = self.trainML(train, test, X_train, X_test, y_train, y_test, model, group, target, id_vars)

                row['fold'] = i
                predictions = predictions.append(pred)
                errors = errors.append(row,ignore_index=True)
                
                if self.hyper_tune:
                    best_params = clf.best_params_
                    best_params_all = best_params_all.append(pd.Series(best_params),ignore_index=True)

                i = i + 1
                if i == n_fold:
                    i = 0
                # print(row)
            
            errors_summary = errors.groupby(['target','model']).mean().reset_index()  
            
            if self.hyper_tune:
                # get best params: mode of each parameters
                best_params_sel = best_params_all.mode().iloc[0,:].to_dict()
                # print(best_params_all, best_params_sel)

                if 'n_estimators' in best_params_sel.keys():
                    # n_estimators must be an integer
                    best_params_sel['n_estimators'] = int(best_params_sel['n_estimators'])
                if 'min_samples_leaf' in best_params_sel.keys():
                    if best_params_sel['min_samples_leaf']>0.5:
                        best_params_sel['min_samples_leaf'] = int(best_params_sel['min_samples_leaf'])
                if 'min_samples_split' in best_params_sel.keys():
                    if best_params_sel['min_samples_split']>1.0:
                        best_params_sel['min_samples_split'] = int(best_params_sel['min_samples_split'])

                self.best_params = best_params_sel
                clf = clf.estimator
                clf.set_params(**best_params_sel)
                # print(clf)

            # re-train model with all training data
            clf.fit(X,y)
            
            self.clf = clf
            self.pred = data[id_vars+[target]].copy()
            self.pred.loc[:,'_pred'] = clf.predict(X)
            
            if self.hybrid_model:
                pred_hybrid = self.predict_hybrid(data=data)
                pred_hybrid.index = self.pred.index
                self.pred = pd.concat([self.pred,pred_hybrid],axis=1)
                # for col in pred_hybrid.columns:
                #    self.pred.loc[:,col] = pred_hybrid[col]
                
            # self.pred = pred_hybrid
            # self.pred_hybrid = pred_hybrid
            
            self.pred_cv = predictions
            
        else:
            row, pred, clf = self.trainML(data, data, X, X, y, y, model, group, target, id_vars)
            errors = pd.DataFrame().append(row, ignore_index=True)
            predictions = pred
            errors_summary = errors
            self.clf = clf
            self.pred = predictions
        
        predictions['model_name'] = name
        errors['model_name'] = name
        errors_summary['model_name'] = name
        
        return errors, errors_summary, predictions
    
    def predict(self, clf=None, data=None, fix_CO2=False):
        """
        Use trained model to predict
        """
        if clf is None:
            clf = self.clf
            
        if clf is None:
            print('Error: no trained model available')
            return 0
                
        if data is None:
            data1 = self.data.copy()
        else:
            data1 = data.copy()
            
        if fix_CO2: # fix CO2 to 2001 level: 370.09625
            data1['CO2_concentration'] = mcf.co2_2001
        
        X = data1[self.config['features']]

        pred = clf.predict(X)
        
        if (self.parallel_predict) & (isinstance(pred,dd.DataFrame) or isinstance(pred,da.Array)):
            print('dask computing ...',flush=True)
            pred = pred.compute()
        elif self.parallel_predict:
            print('X is not a dask dataframe or dataarray, will not predict with dask parallelization')
        
        return pred
    
    
    def predict_hybrid(self, clf=None, data=None):
        """
        Predict hybrid model results
        """
        if clf is None:
            clf = self.clf
            
        if clf is None:
            print('Error: no trained model available')
            return 0
        
        if data is None:
            data1 = self.data.copy()
        else:
            data1 = data.copy()
        
        Tair = data1['Tmean']-273.15
        co2 = data1['CO2_concentration']
        
        if isinstance(data1, dd.DataFrame):
            Tair = Tair.compute()
            co2 = co2.compute()

        pred = pd.DataFrame()
        pred['_pred_hybrid_ref'] = self.predict(clf=clf, data=data1, fix_CO2=True)
        pred['CO2_effect'] = self.getCO2effect_simple(Tair,co2).values
        pred['_pred_hybrid'] = pred['_pred_hybrid_ref'] * pred['CO2_effect']
        
        return pred
        
    def predict_all(self, data, id_vars=None):
        """
        Predict for new data
        """
        if self.clf is None:
            print('Error: no trained model available')
            return 0
        
        data1 = data.copy()            
            
        if self.config['model'] != 'xgboost':
            data1 = data1.dropna(subset = self.config['features'])
            row_less = data.shape[0] - data1.shape[0]
            if isinstance(data, dd.DataFrame):
                row_less = row_less.compute()
            print(f'dropped {row_less} rows containing missing values.')
        
        if id_vars:
            pred = data1[id_vars].copy()
        else:
            pred = data1.copy()

        if isinstance(pred, dd.DataFrame):
            pred = pred.compute()
        
        # get the ml prediction
        pred['_pred'] = self.predict(data=data1, fix_CO2=False)
                                          
        # get hybrid model prediction
        if self.hybrid_model:
            pred_hybrid = self.predict_hybrid(data=data1)
            pred_hybrid.index = pred.index
            pred = pd.concat([pred, pred_hybrid],axis=1)
            
        pred['model_name'] = self.config['name']
        
        return pred
        
    
    def getCO2effect_simple(self, airT, co2):
        """
        Simple CO2 fertilization effects based on a constant ci/ca value
        K-model 
        From Keenan et al., 2021
        """

        # constant
        R = 8.314  # molar gas constant J mol-1 K-1
        r25 = 42.75 # photorespiration point at 25C, converted to ppm
        deltaH = 37830 # activation energy for gamma star J mol-1
        co2_ref = mcf.co2_2001 # reference CO2 concentration in 2001: average of MLO and SPO: 370.1; MLO: 371.32
        xi = 0.7 # cica = 0.7

        #  calculate gamma star
        expF = (airT - 25) / (R * (airT + 273.15) * 298.15)
        gstar = r25 * np.exp(deltaH * expF)

        # calculate the CO2 scalar
        # this scales GPP to present CO2 to a reference CO2 (2001 value)
        ciRef = xi * co2_ref
        CO2scalarRef = (ciRef - gstar) / (ciRef + 2 * gstar)
        ci = xi * co2
        CO2scalar = (ci - gstar) / (ci + 2 * gstar)

        CO2effect = 1 + (CO2scalar - CO2scalarRef) / CO2scalarRef

        return CO2effect