# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 16:45:17 2016

@author: bishwarup
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder

print "Loading data .... "
os.chdir("/home/bishwarup/Kaggle/Prudential")

train = pd.read_csv("Data/train.csv")
test = pd.read_csv("Data/test.csv")
fold_ids = pd.read_csv("ensemble2/validation_id.csv")


le = LabelEncoder()
le.fit(list(train["Product_Info_2"].unique()) + list(test["Product_Info_2"].unique()))
train["Product_Info_2"] = le.transform(train["Product_Info_2"])
test["Product_Info_2"] = le.transform(test["Product_Info_2"])

train.fillna(-1, inplace = True)
test.fillna(-1, inplace = True)

feature_names = [x for x in train.columns if x not in ["Id", "Response", "trainFlag"]]
test_X = np.matrix(test[feature_names])

gbm_val = pd.DataFrame(columns = ["Id", "ground_truth", "gbm_preds"])
gbm_test = pd.DataFrame(dict({"Id" : test["Id"]}))

for i in xrange(10):

    print "\n--------------------------------------------"
    print "------------- Fold %d -----------------------" %i
    print "--------------------------------------------"
    
    val_ids = fold_ids.ix[:, i].dropna()
    idx = train["Id"].isin(list(val_ids))
    
    trainingSet = train[~idx]
    validationSet = train[idx]
    
    tr_X = np.matrix(trainingSet[feature_names])
    tr_Y = np.array(trainingSet["Response"])
    val_X = np.matrix(validationSet[feature_names])
    val_Y = np.array(validationSet["Response"])
    
    gbm = GradientBoostingRegressor(loss= "ls",
                                    learning_rate = 0.06,
                                    max_depth = 7,
                                    n_estimators = 1400,
                                    max_features = 0.82,
                                    min_samples_split = 12,
                                    verbose = 2,
                                    random_state = 112)
    
    gbm.fit(tr_X, tr_Y)                          
    preds = gbm.predict(val_X)
    df = pd.DataFrame(dict({"Id" : validationSet["Id"], "ground_truth" : validationSet["Response"], 
                            "gbm_preds" : preds}))
    gbm_val = gbm_val.append(df, ignore_index = True)
    tpreds = gbm.predict(test_X)
    cname = "Fold" + `i`
    gbm_test[cname] = tpreds
    

gbm_val.to_csv("ensemble2/gbm_val.csv", index = False)
gbm_test.to_csv("ensemble2/gbm_test.csv", index = False)
