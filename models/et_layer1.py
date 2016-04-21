# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 22:03:07 2016

@author: bishwarup
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 00:59:28 2016

@author: bishwarup
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
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

et_val = pd.DataFrame(columns = ["Id", "ground_truth", "et_preds"])
et_test = pd.DataFrame(dict({"Id" : test["Id"]}))

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
    
    et = ExtraTreesRegressor(n_estimators = 2000, criterion = "mse", max_features = 0.8,
                               max_depth = 22, min_samples_split = 12,
                               n_jobs = -1, random_state = 133, verbose = 1)
    
    et.fit(tr_X, tr_Y)                          
    preds = et.predict(val_X)
    df = pd.DataFrame(dict({"Id" : validationSet["Id"], "ground_truth" : validationSet["Response"], 
                            "et_preds" : preds}))
    et_val = et_val.append(df, ignore_index = True)
    tpreds = et.predict(test_X)
    cname = "Fold" + `i`
    et_test[cname] = tpreds
    

et_val.to_csv("ensemble2/et_val.csv", index = False)
et_test.to_csv("ensemble2/et_test.csv", index = False)

    