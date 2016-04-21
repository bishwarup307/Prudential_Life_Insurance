# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 03:53:51 2016

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

train["train_flag"] = 1
test["train_flag"] = 0
test["Response"] = -1

alldata = train.append(test, ignore_index = True)

exclude_cols = ["Id", "Response", "train_flag", "Product_Info_2"]
selected_cols = [x for x in train.columns if x not in exclude_cols]

alldata["na_count"] = alldata[selected_cols].apply(lambda x: x.isnull().sum(), axis = 1)
alldata["zero_count"] = alldata[selected_cols].apply(lambda x: x.value_counts().get(0, 0), axis = 1)
alldata["int_1"] = alldata["BMI"] * alldata["Ins_Age"]

le = LabelEncoder()
alldata["Product_Info_2"] = le.fit_transform(alldata["Product_Info_2"])
alldata.fillna(-1, inplace = True)

ptr = alldata[alldata["train_flag"] == 1]
pte = alldata[alldata["train_flag"] == 0]

feature_names = [x for x in ptr.columns if x not in ["Id", "Response", "train_flag"]]
test_X = np.matrix(pte[feature_names])

eval_matrix = pd.DataFrame(columns = ["Fold", "Id", "ground_truth", "rf2_preds"])
test_matrix = pd.DataFrame({"Id" : pte["Id"]})


for i in xrange(1,10):

    print "\n--------------------------------------------"
    print "---------------- Fold %d --------------------" %i
    print "--------------------------------------------"
    
    val_ids = fold_ids.ix[:, i].dropna()
    idx = ptr["Id"].isin(list(val_ids))
    
    trainingSet = ptr[~idx]
    validationSet = ptr[idx]
    
    tr_X = np.matrix(trainingSet[feature_names])
    tr_Y = np.array(trainingSet["Response"])
    val_X = np.matrix(validationSet[feature_names])
    val_Y = np.array(validationSet["Response"])
    
    rf = RandomForestRegressor(n_estimators = 1500, criterion = "mse", max_features = 0.75,
                               max_depth = 30, min_samples_split = 40, bootstrap= False,
                               n_jobs = -1, random_state = 133, verbose = 1)
    
    rf.fit(tr_X, tr_Y)                          
    preds = rf.predict(val_X)
    df = pd.DataFrame({"Fold" : np.repeat((i + 1), validationSet.shape[0]) ,"Id" : validationSet["Id"], "ground_truth" : validationSet["Response"], 
                            "rf2_preds" : preds})

    eval_matrix = eval_matrix.append(df, ignore_index = True)
    tpreds = rf.predict(test_X)
    cname = "Fold" + `i+1`
    test_matrix[cname] = tpreds
    

eval_matrix.to_csv("ensemble2/rf2_eval.csv", index = False)
test_matrix.to_csv("ensemble2/rf2_test.csv", index = False)

    