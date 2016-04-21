# -*- coding: utf-8 -*-
"""
Kaggle: Prudential Life Insurance

@author: bishwarup
"""

import os
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adadelta
from keras.regularizers import l1, l2, l1l2

print "Loading data .... "
os.chdir("/home/bishwarup/Kaggle/Prudential")

md = pd.read_csv("Data/NN_processed.csv")
fold_id = pd.read_csv("ensemble2/validation_id.csv")

ptr = md[md.split1 == 0].copy()
pte = md[md.split1 == 2].copy()

feature_names = [x for x in ptr.columns if x not in ["Id", "Response", "split1"]]
nnVal = pd.DataFrame(columns = ["Id", "ground_truth", "nnPreds"])
nn_test_matrix = pd.DataFrame(dict({"Id" : pte["Id"]}))

for i in xrange(10):
    
    print "\n--------------------------------------------"
    print "----------- Fold %d --------------------" %i
    print "--------------------------------------------"
    
    val_ids = fold_id.ix[:, i].dropna()
    idx = ptr["Id"].isin(list(val_ids))
    
    trainingSet = ptr[~idx]
    validationSet = ptr[idx]
    
    tr_Y = np.array(trainingSet["Response"].copy())
    val_Y = np.array(validationSet["Response"].copy())
    
    tr_X = np.matrix(trainingSet[feature_names])
    val_X = np.matrix(validationSet[feature_names])
    te_X = np.matrix(pte[feature_names])
    
    model = Sequential()
    model.add(Dense(21, input_dim = 559, init = 'glorot_normal', activation='tanh', W_regularizer=l1l2(l1 = 1e-4, l2 = 1e-4)))
    model.add(Dropout(0.15))
    model.add(Dense(1, init = 'zero', activation='linear'))
    model.compile(loss = "mse", optimizer = "adagrad")
    
    model.fit(tr_X, tr_Y, nb_epoch=200, batch_size=25, validation_data= (val_X, val_Y))
    
    preds = model.predict(val_X, batch_size=128).ravel()
    df = pd.DataFrame(dict({"Id" : validationSet["Id"], "ground_truth" : validationSet["Response"], 
                            "nnPreds" : preds}))
    nnVal = nnVal.append(df, ignore_index = True)
    
    tpreds = model.predict(te_X, batch_size=128).ravel()
    cname = "Fold" + `i`
    nn_test_matrix[cname] = tpreds
    
    
nnVal.to_csv("ensemble2/nn1val.csv", index = False)
nn_test_matrix.to_csv("ensemble2/nn1test.csv", index = False)