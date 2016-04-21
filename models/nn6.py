# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 14:17:37 2016

@author: bishwarup
"""


import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import theano
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.optimizers import Adagrad,SGD,Adadelta
from keras.regularizers import l1, l2, l1l2
from keras.callbacks import Callback

os.chdir("/home/bishwarup/Kaggle/Prudential")
np.random.seed(133)
nb_epoch = 250
batch_size = 28
n_class = 1


def load_data():

    print "\nLoading..."
    alldata = pd.read_csv("Data/alldata_log.csv")
    
    cat_cols = [f for f in alldata.columns if f.startswith("Product_Info_2")]    
    cat_df = alldata[cat_cols].copy()
    cat_df[cat_df == 0] = -1

    alldata.drop(cat_cols, axis = 1, inplace = True)
    alldata = pd.concat([alldata, cat_df], axis = 1)
    
    ptr = alldata[alldata["trainFlag"] == 1]
    pte = alldata[alldata["trainFlag"] == 0]
    
    return ptr, pte
    

def build_model(train):
    
    input_dim = train.shape[1]
    model = Sequential()
    model.add(Dense(15, input_dim = input_dim, init = 'glorot_normal', activation='tanh', W_regularizer=l1l2(l1 = 1e-4, l2 = 1e-4)))
    model.add(Dropout(0.15))
    model.add(Dense(1, init = 'zero', activation='linear'))
    model.compile(loss = "mse", optimizer = "adagrad")
    
    return model

def fit_model(ptr, pte):
    
    feature_names = [f for f in ptr.columns if f not in ["Id", "Response", "trainFlag"]]

    print "\nStarting validation...."
    fold_ids = pd.read_csv("ensemble2/validation_id.csv")

    eval_matrix = pd.DataFrame(columns = ["Fold", "Id", "ground_truth", "nn6_preds"])        
    test_matrix = pd.DataFrame({"Id" : pte["Id"]})
    
    for i in xrange(10):
        
        print "\n--------------------------------------------"
        print "---------------- Fold %d --------------------" %i
        print "--------------------------------------------"
        
        val_ids = fold_ids.ix[:, i].dropna()
        idx = ptr["Id"].isin(list(val_ids))
        
        trainingSet = ptr[~idx]
        validationSet = ptr[idx]
        
        tr_Y = np.array(trainingSet["Response"].copy())
        val_Y = np.array(validationSet["Response"].copy())
        
        tr_X = np.matrix(trainingSet[feature_names])
        val_X = np.matrix(validationSet[feature_names])
                
            
            
        model = build_model(tr_X)
        model.fit(tr_X, tr_Y, nb_epoch=nb_epoch, batch_size=batch_size, validation_data= (val_X, val_Y))
        
        preds = model.predict(val_X, batch_size=128).ravel()
        df = pd.DataFrame({"Fold" : np.repeat((i + 1), validationSet.shape[0]) ,"Id" : validationSet["Id"], "ground_truth" : validationSet["Response"], 
                            "nn6_preds" : preds})
        eval_matrix = eval_matrix.append(df, ignore_index = True)
     
     
    tr_X = np.matrix(ptr[feature_names])
    tr_Y = np.array(ptr["Response"].copy())
    te_X = np.matrix(pte[feature_names])

    model = build_model(tr_X)
    model.fit(tr_X, tr_Y, nb_epoch=nb_epoch, batch_size=batch_size)
    tpreds =   model.predict(te_X, batch_size=128).ravel()
    test_matrix["nn6_preds"] = tpreds
    return eval_matrix, test_matrix

#############################

ptr, pte = load_data()
validm, testm = fit_model(ptr, pte)

validm.to_csv("ensemble2/nn5_eval.csv", index = False)
testm.to_csv("ensemble2/nn5_test.csv", index = False)
