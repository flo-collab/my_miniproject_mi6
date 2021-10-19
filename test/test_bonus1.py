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

def test_tree_vs_rfc (): 
    list_tree_number = [1, 4, 8, 20 ]
    list_RFC = []
    for i in list_tree_number :
        list_RFC.append(RandomForestClassifier(n_estimators = i).fit(X_train, y_train))

    DTC = tree.DecisionTreeClassifier().fit(X_train, y_train)
    score_DTC = metrics.accuracy_score(y_test,DTC.predict(X_test))
    list_score_RFC = []
    for model in list_RFC :
        list_score_RFC.append(metrics.accuracy_score(y_test,model.predict(X_test)))
    print(score_DTC)
    print(list_score_RFC)

    for score in list_score_RFC :
        assert score >= score_DTC
