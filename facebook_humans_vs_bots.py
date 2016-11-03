# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 10:32:22 2016

@author: user
"""

import pandas as pd
import re
import gc
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.externals import joblib
import xgboost as xgb
#import xgboost as xgb
bids=pd.read_csv('bids.csv')
#bids = bids.replace({' ': ''}, regex = Tr*ue)
bids=bids.sort_values(['bidder_id','time'], ascending = [True,True])
bidder = pd.DataFrame(data = bids['bidder_id'].unique(), columns = ['bidder_id'],
                    index = bids['bidder_id'].unique())
count=bids.groupby('bidder_id')['bidder_id'].agg('count')
bidder['count']=count
print(bids.info())
time_diff=bids.groupby('bidder_id')['time'].diff()
time_diff_str=time_diff.astype(str).fillna('')
bids['time_diff']=time_diff
bids['time_diff_str']=time_diff_str
max_time=bids.groupby('bidder_id')['time'].max()
bidder['max_time']=max_time
min_time=bids.groupby('bidder_id')['time'].min()
bidder['min_time']=min_time
max_diff=bids.groupby('bidder_id')['time_diff'].max()
bidder['max_diff']=max_diff.fillna(max_diff.mean())
min_diff=bids.groupby('bidder_id')['time_diff'].min()
bidder['min_diff']=min_diff.fillna(max_diff.mean())
range=max_diff-min_diff
bidder['range']=range
mean_diff=bids.groupby('bidder_id')['time_diff'].mean()
bidder['mean_diff']=mean_diff.fillna(mean_diff.mean())
no_ip=bids.groupby('bidder_id')['ip'].agg('count')
bidder['ip_count']=no_ip
no_country=bids.groupby('bidder_id')['country'].agg('count')
bidder['country_count']=no_country
no_device=no_country=bids.groupby('bidder_id')['device'].agg('count')
bidder['device_count']=no_device
no_auctions=bids.groupby('bidder_id')['auction'].agg('count')
bidder['auction_count']=no_auctions
no_merchandise=bids.groupby('bidder_id')['merchandise'].agg('count')
bidder['merchandise_count']=no_merchandise
#no_bids_per_auction=bids.groupby('count')['device'].agg('count')
print(bidder.head())
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
combo=train.append(test)
combo=combo.merge(bidder,how='left',left_on=['bidder_id'],right_on=['bidder_id'])
del train
del test
del bids
del bidder
gc.collect();
print(combo.head())
col=['merchandise_count','auction_count','device_count','country_count','ip_count','mean_diff','range','min_diff','max_diff','count','max_time','min_time']

for i in col:
    x=combo.groupby('outcome')[i].mean()
    print(x)
    
train=combo.ix[0:4700]   
xtest=combo.ix[4701:]
y=train['outcome']
xtrain=train.drop(['outcome'],axis=1)
feats_25 = SelectPercentile(chi2, 25).fit(xtrain, y)
xtrain = feats_25.transform(xtrain)
xtest = feats_25.transform(xtest)

clf = xgb.XGBClassifier(objective = 'binary:logistic',
                            learning_rate = 0.05,
                            max_depth = 5,
                            nthread = 8,
                            seed = 42,
                            subsample = 0.4,
                            colsample_bytree = 0.7,
                            min_child_weight = 1,
                            n_estimators = 100,
                            gamma = 0.15, silent = True)

#bag of 15 models
rounds = 15
sample = pd.read_csv('sampleSubmission.csv')
preds_mat = np.zeros((len(sample.index), rounds))
for i in range(rounds):
    clf.set_params(seed = i + 1)
    clf.fit(xtrain, y)
    preds_tmp = clf.predict_proba(xtest)[:, 1]
    preds_mat[:, i] = preds_tmp
bagged_preds = preds_mat.mean(axis = 1)
sample.prediction = bagged_preds
sample.to_csv('submissions/facebook_submission.csv', index = False)
    