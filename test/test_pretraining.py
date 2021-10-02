import pytest
import pandas as pd
import numpy as np
from src.gini import gini 
from src.gain import gain 
from src.RFCdepth import RF_accuracy  
import pickle
from src.model import  X_train, data , X_test, y_train, y_test, X, y
from sklearn.metrics import  accuracy_score
from sklearn.ensemble import RandomForestClassifier


def test_gini():
    assert round(gini(data.Survived), 3) == 0.473
    assert round(gini(pd.Series([1, 1, 1, 1, 1, 1, 1, 1])), 3) == 0
    assert round(gini(pd.Series([1, 1, 1, 1, 1, 1, 1, 0])), 3) == 0.219
    assert round(gini(pd.Series([1, 1, 1, 1, 1, 1, 0, 0])), 3) == 0.375
    assert round(gini(pd.Series([1, 1, 1, 1, 1, 0, 0, 0])), 3) == 0.469
    assert round(gini(pd.Series([1, 1, 1, 1, 0, 0, 0, 0])), 3) == 0.500

def test_gain():
    assert round(gain(data, "Survived", "Pclass"), 3) == 0.055
    assert round(gain(data, "Survived", "Sex"), 3) == 0.14
    
def test_check():
    model = pickle.load(open('../src/model.pkl', 'rb'))
    sortie=model.predict(X_train)
    
    assert sortie.shape[0] == X_train.shape[0]

def test_proba():
    model = pickle.load(open('../src/model.pkl', 'rb'))
    proba = model.predict_proba(X_train)
    pb=pd.DataFrame(proba, columns=['yes','no'])
    sum_pb = pb['yes']+pb['no']
    
    assert [pb[col].between(0, 1, inclusive=True).any() for col in pb.columns]
    assert [sum_pb.between(0, 1, inclusive=True).any()]
    assert (proba <= 1).all() & (proba >= 0).all()

def test_holdout():
    test_size = round(X_test.shape[0] / X.shape[0],1)
    train_size = round(X_train.shape[0] / X.shape[0],2)
    
    assert test_size == 0.2
    assert train_size == 0.8
    assert X_train.shape[1] == X_test.shape[1]
    assert (y_train.shape[0] + y_test.shape[0]) == y.shape[0]
    assert (list(X_test.columns)) == (list(X_train.columns))
    
def test_overfitRTC():  
    rfc = pickle.load(open('../src/model.pkl', 'rb'))
    pred_train = np.round(rfc.predict(X_train))
    pred_test = np.round(rfc.predict(X_test))
    acc_train = accuracy_score(y_train, pred_train)
    acc_test = accuracy_score(y_test, pred_test)
    
    assert acc_train > acc_test
    assert (acc_train - acc_test) / acc_train <= 0.1


def test_RF_accuracy():    
    assert  [RF_accuracy(i)<RF_accuracy(i+1) for i in range (4,8)]