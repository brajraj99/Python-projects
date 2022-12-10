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


# -----------------------------------------------------------------------------
# Calling the function dataset() to read the csv data file and obtaining the
# original data frame and transposed dataframe
df, df_tran = dataset('API_19_DS2_en_csv_v2_4700503')

# changing the header of the transposed datafram and populating it with the
# country names
header1 = df_tran.iloc[0]
df_tran = df_tran[1:]
df_tran.columns = header1

# Extracting information from the transposed matrix for different countries
# Creating a dataframe for India and changing the column names of that
# dataframe as the Indicator name
df_india = df_tran['India']
# The header extracted below can be used for headers of different countries
header_country = df_india.iloc[1]
# Renaming the columns as Indicator names
df_india.columns = header_country
# Dropping the unwanted rows from the dataframe
df_india = df_india.drop(['Indicator Name', 'Country Code', 'Indicator Code'])
# Likewise we can create dataframes of United States
df_usa = df_tran['United States']
df_usa.columns = header_country
df_usa = df_usa.drop(['Indicator Name', 'Country Code', 'Indicator Code'])
# Likewise we can create dataframes of United States
df_somalia = df_tran['Somalia']
df_somalia.columns = header_country
df_somalia = df_somalia.drop(
    ['Indicator Name', 'Country Code', 'Indicator Code'])
