import pytest
import pandas as pd
import numpy as np
import pickle
from model import  X_test



def test_sex():
    rfc = pickle.load(open('model.pkl', 'rb'))
    df_female = X_test.copy()
    df_male = X_test.copy()
    df_female["Sex"]=1
    df_male["Sex"]=0
    prob_female = rfc.predict_proba(df_female)[:, 1]
    prob_male = rfc.predict_proba(df_male)[:, 1]
    nb_female = np.count_nonzero((prob_female > 0.5))
    nb_male = np.count_nonzero((prob_male > 0.5))
    assert nb_female  >= nb_male
    assert (nb_female - nb_male)/(nb_female + nb_male) > 90/100
    assert [prob_female[i] >= prob_male[i] for i in range(prob_female.shape[0])]
    

def test_fare():
    rfc = pickle.load(open('../src/model.pkl', 'rb'))
    df_fare = X_test.copy()
    df_fare["Fare"]==8.5
    fare8 = rfc.predict_proba(df_fare)[:, 1]
    df_fare['Fare'] = 26
    fare26 = rfc.predict_proba(df_fare)[:, 1]
    df_fare['Fare'] = 13
    fare13 = rfc.predict_proba(df_fare)[:, 1]
    nb_8 = np.count_nonzero((fare8 > 0.5))
    nb_26 = np.count_nonzero((fare26 > 0.5))
    nb_13 = np.count_nonzero((fare13 > 0.5))
    assert nb_8  >= nb_26
    assert nb_13 >= nb_26
    assert nb_8 >= nb_13
