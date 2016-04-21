# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 19:17:11 2016

@author: bishwarup
"""

import os
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVR

print "Loading data .... "
os.chdir("/home/bishwarup/Kaggle/Prudential")

train = pd.read_csv("Data/svm_train.csv")
test = pd.read_csv("Data/svm_test.csv")
fold_ids = pd.read_csv("ensemble2/validation_id.csv")

train.fillna(-1, inplace = True)
test.fillna(-1, inplace = True)

linsvr_val = pd.DataFrame(columns = ["Id", "ground_truth", "linsvr_preds"])
linsvr_test = pd.DataFrame(dict({"Id" : test["Id"]}))

feature_names = [x for x in train.columns if x not in ["Id", "Response", "trainFlag"]]
test_X = np.matrix(test[feature_names])

for i in xrange(10):
    
    print "\n--------------------------------------------"
    print "----------- Fold %d -----------------------" %i
    print "--------------------------------------------"
    
    val_id = fold_ids.ix[:, i].dropna()
    idx = train["Id"].isin(list(val_id))
    
    trainingSet = train[~idx]
    validationSet = train[idx]
    
    tr_X = np.matrix(trainingSet[feature_names])
    tr_Y = np.array(trainingSet["Response"])
    val_X = np.matrix(validationSet[feature_names])
    val_Y = np.array(validationSet["Response"])
    
    regm = LinearSVR(C = 0.06, epsilon = 0.45, tol = 1e-5,
                     dual = True, verbose = True, random_state = 133)
                     
    regm.fit(tr_X, tr_Y)    
    preds = regm.predict(val_X)
    
    df = pd.DataFrame(dict({"Id" : validationSet["Id"], "ground_truth" : validationSet["Response"], 
                            "linsvr_preds" : preds}))
    
    linsvr_val = linsvr_val.append(df, ignore_index = True)
    
    tpreds = regm.predict(test_X)
    cname = "Fold" + `i`
    linsvr_test[cname] = tpreds
    
linsvr_val.to_csv("ensemble2/linsvr_val.csv")
linsvr_test.to_csv("ensemble2/linsvr_test.csv")