# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 12:39:45 2016

@author: bishwarup
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso, Ridge

print "Loading data .... "
os.chdir("/home/bishwarup/Kaggle/Prudential")

train = pd.read_csv("Data/svm_train.csv")
test = pd.read_csv("Data/svm_test.csv")
fold_ids = pd.read_csv("ensemble2/validation_id.csv")

train.fillna(-1, inplace = True)
test.fillna(-1, inplace = True)

lasso_val = pd.DataFrame(columns = ["Id", "ground_truth", "lasso_preds"])
ridge_val = pd.DataFrame(columns = ["Id", "ground_truth", "ridge_preds"])

feature_names = [x for x in train.columns if x not in ["Id", "Response", "trainFlag"]]
test_X = np.matrix(test[feature_names])

for i in xrange(4):
    
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
    
    las = Lasso(alpha = 0.0005,
            copy_X = True,
            tol = 1e-6,
            selection = "random",
            random_state = 112,
            max_iter = 2000)
            
    rd = Ridge(alpha = 1e-6,
           copy_X = True,
           tol = 1e-7,
           max_iter = 2000)
           
    las.fit(tr_X, tr_Y)
    rd.fit(tr_X, tr_Y)
    
    lpreds = las.predict(val_X)
    rpreds = rd.predict(val_X)
    
    df_lasso = pd.DataFrame(dict({"Id" : validationSet["Id"], "ground_truth" : validationSet["Response"], 
                            "lasso_preds" : lpreds}))

    df_ridge = pd.DataFrame(dict({"Id" : validationSet["Id"], "ground_truth" : validationSet["Response"], 
                                "ridge_preds" : rpreds}))
    
    lasso_val = lasso_val.append(df_lasso, ignore_index = True)
    ridge_val = ridge_val.append(df_ridge, ignore_index = True)

lasso_val.to_csv("ensemble2/lasso_val.csv", index = False) 
ridge_val.to_csv("ensemble2/ridge_val.csv", index = False) 

tr_X = np.matrix(train[feature_names])               
tr_Y = np.array(train["Response"])

las = Lasso(alpha = 0.0005,
        copy_X = True,
        tol = 1e-6,
        selection = "random",
        random_state = 112,
        max_iter = 2000)
        
rd = Ridge(alpha = 1e-6,
       copy_X = True,
       tol = 1e-7,
       max_iter = 2000)
       
las.fit(tr_X, tr_Y)
lpreds = las.predict(test_X)
lasso_test = pd.DataFrame({"Id" : test["Id"], "lasso_preds" : lpreds})
lasso_test.to_csv("ensemble2/lasso_test.csv", index = False)

rd.fit(tr_X, tr_Y)
rpreds = rd.predict(test_X)
ridge_test = pd.DataFrame({"Id" : test["Id"], "ridge_preds" : rpreds})
ridge_test.to_csv("ensemble2/ridge_test.csv", index = False)
