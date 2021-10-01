#Data preparation approach for conti attributes.
import pandas as pd


def impute_numeric(df: pd.DataFrame, col: str, impute_type: str = 'median') -> pd.DataFrame:
   
    if impute_type == 'median':
        impute_value = df[col].median()  # type: float
    elif impute_type == 'mean':
        impute_value = df[col].mean()
    elif impute_type == '-1':
        impute_value = -1
    else:
        raise NotImplementedError('recognize impute_type options "mean", "median", "-1')

    df.loc[df[col].isnull(), col] = impute_value
    return df