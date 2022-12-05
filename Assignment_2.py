#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 11:24:38 2022

@author: risingphoenix
"""

import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np


def dataset(filename):
    """
    
    The function takes in a filename as argument, reads a dataframe in World - 
    bank format and returns two dataframes: one with years as columns and one 
    with countries as columns.

    Parameters
    ----------
    filename : File name or directory of the dataset.

    Returns
    -------
    The function returns the original dataframe and the transposed dataframe.

    """
    
    df_a = pd.read_csv(f"{filename}.csv")   
    df_b = df_a.transpose()
    
    return df_a, df_b
