# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 17:55:19 2016

@author: bishwarup
"""
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
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

rf_val2 = pd.DataFrame(columns = ["Id", "ground_truth", "rf_preds"])
rf_test2 = pd.DataFrame(dict({"Id" : test["Id"]}))

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
    
    rf = RandomForestRegressor(n_estimators = 2000, criterion = "mse", max_features = 0.8,
                               max_depth = 25, min_samples_split = 20,
                               n_jobs = -1, random_state = 133, verbose = 1)
    
    rf.fit(tr_X, tr_Y)                          
    preds = rf.predict(val_X)
    df = pd.DataFrame(dict({"Id" : validationSet["Id"], "ground_truth" : validationSet["Response"], 
                            "rf_preds" : preds}))
    rf_val2 = rf_val2.append(df, ignore_index = True)
    tpreds = rf.predict(test_X)
    cname = "Fold" + `i`
    rf_test2[cname] = tpreds
    

rf_val2.to_csv("ensemble2/rf_val2.csv", index = False)
rf_test2.to_csv("ensemble2/rf_test2.csv", index = False)

    