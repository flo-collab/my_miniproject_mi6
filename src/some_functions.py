import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

def list_proba(df_one_col:pd.Series):
    return (df_one_col.value_counts()/df_one_col.shape[0])

def test_list_proba(df):
    for col in df.columns :
        assert list_proba(df[col].notnull()).values.sum()==1


def gini (df_one_col:pd.Series):
    gini = 1
    for i in (df_one_col.value_counts()/df_one_col.shape[0]).values :
        gini = gini - i*i
    return gini

def entropy(df_one_col:pd.Series):
    entropy = 0
    for p in list_proba(df_one_col).values :
        entropy = entropy - p*math.log2(p)
    return entropy

def get_entropy_df(df,col1,col2):
    df_compo_entropy = df.pivot_table('PassengerId', index=col2, columns=col1,aggfunc='count',margins=True,margins_name="Total")
    df_compo_entropy.drop(['Total'],axis=0,inplace =True)
    return df_compo_entropy