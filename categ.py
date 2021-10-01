#Data preparation approach for categ attributes.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pytest 


def smalletter_string(string: str) -> str:
  
    if isinstance(string, str):
        return string.lower()
    return None


def smalletter_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
   
    df[col] = df[col].apply(smalletter_string)
    return df


def ext_status(df: pd.DataFrame, col: str, replace_dict: dict = None,
                  status_col: str = 'title') -> pd.DataFrame:
    
    df[status_col] = df[col].str.extract(r' ([A-Za-z]+)\.', expand=False)

    if replace_dict:
        df[status_col] = np.where(df[status_col].isin(replace_dict.keys()),
                                 df[status_col].map(replace_dict),
                                 df[status_col])

    return df