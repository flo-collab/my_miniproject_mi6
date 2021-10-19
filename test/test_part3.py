from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
from src.model import X_train, X_test, y_train, y_test
from src.model import X, y



def test_decrease_score():
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X, y, test_size=0.05, random_state=0)

    DTC = tree.DecisionTreeClassifier().fit(X_train, y_train)
    score_DTC = metrics.accuracy_score(y_test,DTC.predict(X_test))
    
    DTC_new = tree.DecisionTreeClassifier().fit(X_train_new, y_train_new)
    score_DTC_new = metrics.accuracy_score(y_test_new,DTC_new.predict(X_test_new))
    assert score_DTC_new >= score_DTC

