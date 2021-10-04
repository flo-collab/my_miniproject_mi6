import pytest
import numpy as np
from dt import DecisionTree
from rf import RandomForest

from dtp import load_df, df_ready, df_divide, get_attlab

from dt import DecisionTree
from rf import RandomForest


#Prevent repeating code in your unit tests with Fixtures
@pytest.fixture
def t_d():
    df = load_df()
    df = df_ready(df)

    train, test = df_divide(df)
    X_train, y_train = get_attlab(train)
    X_test, y_test = get_attlab(test)
    return X_train, y_train, X_test, y_test

@pytest.fixture
def d_tit():
    df = load_df()
    df.columns = [col.lower() for col in df.columns]

    train, test = df_divide(df)
    return train, test

#Prevent repeating code in your unit tests with Fixtures
@pytest.fixture
def p_d():
    # cheap 13th pclass13 male
    passenger13 = {'PassengerId': 13,
                  'Survived': None,
                  'Pclass': 3,
                  'Name': ' Mr. William',
                  'Sex': 'male',
                  'Age': 20.0,
                  'SibSp': 0,
                  'Parch': 0,
                  'Ticket': 'A/5. 2151',
                  'Fare': 8.05,
                  'Cabin': None,
                  'Embarked': 'S'}

    # expensive 2nd pclass2 female)
    passenger2 = {'PassengerId': 2,
                  'Survived': None,
                  'Pclass': 1,
                  'Name': ' Mrs. John',
                  'Sex': 'female',
                  'Age': 38.0,
                  'SibSp': 1,
                  'Parch': 0,
                  'Ticket': 'PC 17599',
                  'Fare': 71.2833,
                  'Cabin': 'C85',
                  'Embarked': 'C'}

    return passenger13, passenger2

#Prevent repeating code in your unit tests with Fixtures
@pytest.fixture
def tdt_tit(t_d):
    X_train, y_train, _, _ = t_d
    dt = DecisionTree(depth_limit=5)
    dt.fit(X_train, y_train)
    return dt

#Prevent repeating code in your unit tests with Fixtures
@pytest.fixture
def rfd_tit(t_d):
    X_train, y_train, _, _ = t_d
    rf = RandomForest(num_trees=8, depth_limit=5, col_subsampling=0.8, row_subsampling=0.8)
    rf.fit(X_train, y_train)
    return rf



