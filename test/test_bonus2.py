from sklearn import tree
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
from src.model import X_train, X_test, y_train, y_test

list_deepness = [i for i in range(1, 11)]
print(list_deepness)

#créer un test qui vérifie que pour une même profondeur, 
# le randomforest ( avec 7 arbres par exemple) est plus précis qu’un arbre de décision de même profondeur
deepness = 4
def test_dtc_vs_rfc():
    DTC = tree.DecisionTreeClassifier(max_depth = deepness).fit(X_train, y_train)
    RFC = RandomForestClassifier(n_estimators = 7 ,max_depth = deepness).fit(X_train, y_train)
    score_DTC = metrics.accuracy_score(y_test,DTC.predict(X_test))
    score_RFC = metrics.accuracy_score(y_test,RFC.predict(X_test))
    assert score_RFC >= score_DTC

