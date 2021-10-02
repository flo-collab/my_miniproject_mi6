import pandas as pd
import numpy as np


def gini(node):
    p = node.value_counts()/node.shape[0]
    gini = 1-np.sum(p**2)
    return(round(gini, 3))