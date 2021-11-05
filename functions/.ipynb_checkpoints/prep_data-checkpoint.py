import numpy as np
from math import sqrt
import pandas as pd

import sklearn 
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from sklearn.feature_selection import SelectFromModel
import xgboost
from xgboost import XGBRegressor
#functions

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# transform a time series dataset into a supervised learning dataset
'''def series_to_supervised(data, target, n_in=1, n_out=1, dropnan=True):
    """
    Parameters
    ---------
    data - original dataset
    target - pandas Series object to predict
    n_in - number of historical lags to include in forecast
    n_out - number of lags predicted in future
    """ 
    df = data.copy()#pd.DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, -1, -1):
        #print(i)
        #cols.append(pd.DataFrame(df.shift(i).values, columns=[col+'-'+str(i) for col in df.columns]))
        df_shift=df.shift(i)
        df_shift.columns=[col+'-'+str(i) for col in df_shift.columns]
        cols.append(df_shift)#df.shift(i))
    
    # forecast n_out step
    cols.append(target.shift(-n_out))

    # put it all together
    agg = pd.concat(cols, axis=1)

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    """
     Returns
    -------
    agg: pd.DataFrame with  with lagged features in columns where column name expresses the feature and lag, 
    e.g. close-3 means close value 3 days ago
    """ 
    return agg#.values
'''
# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    return data[:-n_test, :], data[-n_test:, :]

#Find the most important features
#using the probability of influence
'''def features_importance_ratio(model, ratio=0.95):
    valuable_features=[i for i,v in enumerate(model.feature_importances_) if v > np.quantile(model.feature_importances_, ratio)]
    return [x for x in res.columns[valuable_features]]
'''
#using the top N
import operator
'''def features_importance_top_N(model, topN=10):
    dict_features={}
    for i,v in enumerate(model.feature_importances_):
        dict_features[i]=v
    sorted_d = dict( sorted(dict_features.items(), key=operator.itemgetter(1),reverse=True))
    valuable_features=list(sorted_d.keys())[:topN]
    return [x for x in res.columns[valuable_features]]
'''
'''def get_N_features_dict(model, verbose=True):
    thresholds = np.sort(model.feature_importances_)
    feature_factor=dict()
    for thresh in np.unique(thresholds)[-50:]:
        #print(thresh)
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(trainX)
        # train model
        selection_model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        selection_model.fit(select_X_train, trainy)
        # eval model
        select_X_test = selection.transform(testX)
        y_pred = selection_model.predict(select_X_test)
        predictions = [(value) for value in y_pred]
        mae = mean_absolute_error(testy, predictions)
        mape=mean_absolute_percentage_error(testy, predictions)
        if verbose:
          print("Thresh=%.5f, n=%d, MAE: %.3f" % (thresh, select_X_train.shape[1], mae))
        feature_factor[select_X_train.shape[1]]=mae
    #print(min(feature_factor, key=feature_factor.get))
    return feature_factor
'''


#Select features
#Select features excluding date

#create lagged features
# transform a time series dataset into a supervised learning dataset
def lagged_features(df, target, n_in=1, n_out=1, dropnan=True):
    """
    Parameters
    ---------
    df - dataframe with all the freatures  the lagged values are created for
    target - target Series object to predict
    n_in - number of historical lags to include in forecast
    n_out - number of lags predicted in future
    """ 
    cols = list()
    for i in range(n_in, -1, -1):
        df_shift=df.shift(i)
        df_shift.columns=[col+'-'+str(i) for col in df_shift.columns]
        cols.append(df_shift)#df.shift(i))
    
    cols.append(target.shift(-n_out)) # forecast n_out step
    agg = pd.concat(cols, axis=1) # put it all together
    
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    """
     Returns
    -------
    agg: pd.DataFrame with  with lagged features in columns where column name expresses the feature and lag, 
    e.g. close-3 means close value 3 days ago
    """ 
    return agg
import operator

def features_importance_top_N(model, train, topN=10):
    """
    Parameters
    ---------
    model - XGboost model that is tested on feature importance
    topN - alternative number of nyumber of the features that impact the most if there's not much difference in result
    """
    dict_features={}
    for i,v in enumerate(model.feature_importances_):
        dict_features[i]=v
    sorted_d = dict( sorted(dict_features.items(), key=operator.itemgetter(1),reverse=True))
    valuable_features=list(sorted_d.keys())[:topN]
    """
     Returns
    -------
    array of columns that represent the most important features
    """ 
    return [x for x in train.columns[valuable_features]]

def get_N_features_dict(model,trainX,trainy,testX,  testy, verbose=True):
    """
    Parameters
    ---------
    model - XGboost model that is tested on feature importance
    trainX, trainy - datasets the model is trained on to estimate the feature importance
    """
    thresholds = np.sort(model.feature_importances_)
    feature_factor=dict()
    for thresh in np.unique(thresholds)[-50:]:
        #print(thresh)
        # select features using threshold
        selection = SelectFromModel(model, threshold=thresh, prefit=True)
        select_X_train = selection.transform(trainX)
        # train model
        selection_model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
        selection_model.fit(select_X_train, trainy)
        # eval model
        select_X_test = selection.transform(testX)
        y_pred = selection_model.predict(select_X_test)
        predictions = [(value) for value in y_pred]
        mae = mean_absolute_error(testy, predictions)
        if verbose:
          print("Thresh=%.5f, n=%d, MAE: %.3f" % (thresh, select_X_train.shape[1], mae))
        feature_factor[select_X_train.shape[1]]=mae
    """
     Returns
    -------
    feature_factor - dictionary of number of features used and error associated
    """ 
    return feature_factor


