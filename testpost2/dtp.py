#Data prep: Includes train and test split and the creation of attributes and labels.

from typing import Tuple
import pandas as pd
from numpy import array
from sklearn.model_selection import train_test_split
from categ import smalletter_column, ext_status
from conti import impute_numeric

status_mapping = {
    'mme': 'mrs',
    'ms': 'miss',
    'mlle': 'miss',
    'lady': 'rare',
    'countess': 'rare',
    'capt': 'rare',
    'col': 'rare',
    'don': 'rare',
    'dr': 'rare',
    'major': 'rare',
    'sir': 'rare',
    'jonkheer': 'rare',
    'dona': 'rare',
    'the countess': 'rare',
    'rev': 'rare'
}

status_to_int = {'mr': 1, 'mrs': 2, 'miss': 3, 'master': 4, 'rare': 5, 'NA': -1}
sex_to_int = {'female': 1, 'male': 0, 'NA': -1}
port_to_int = {'s': 0, 'c': 1, 'q': 2, 'NA': -1}


def load_df() -> pd.DataFrame:
   
    return pd.read_csv('Titanic.csv', sep=';')


def df_ready(df: pd.DataFrame) -> pd.DataFrame:
    
    # smallcase columns
    df.columns = [col.lower() for col in df.columns]

    # columnsto drop
    df.drop(columns=['passengerid', 'ticket', 'cabin'], inplace=True)

    # construct title column
    df = smalletter_column(df, 'name')
    df = ext_status(df, 'name', status_mapping)
    df.drop(columns=['name'], inplace=True)

    # insert smallcase embarked column
    df = smalletter_column(df, 'embarked')
    df = smalletter_column(df, 'sex')

    # input nulls for the numeric columns
    for col in ['pclass', 'age', 'sibsp', 'parch', 'fare']:
        df = impute_numeric(df, col, '-1')

    # input nan values and categ encoding
    df['title'].fillna('NA', inplace=True)
    df['sex'].fillna('NA', inplace=True)
    df['embarked'].fillna('NA', inplace=True)

    df['title'] = df['title'].map(status_to_int)
    df['sex'] = df['sex'].map(sex_to_int)
    df['embarked'] = df['embarked'].map(port_to_int)

    return df


def df_divide(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
  
    train, test = train_test_split(df, test_size=0.2, random_state=1368, stratify=df['survived'])

    return train, test


def get_attlab(df: pd.DataFrame) -> Tuple[array, array]:
  
    # Get labels and attributes
    X = df.iloc[:, 1:].values
    y = df['survived'].values

    return X, y