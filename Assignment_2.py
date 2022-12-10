#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 11:24:38 2022

@author: risingphoenix
"""

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
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


def heatmap(data, num):
    """
    Function to generate heatmap of Correlation Coefficient Matrix for India
    and Australia

    Parameters
    ----------
    data : TYPE - Dataframe
        Dataframe of Correlation Coefficient Matrix of which heatmap has to be
        made.
    num : TYPE - Int
        1: India
        2: Australia

    Returns
    -------
    None.

    """
    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=(8, 8), dpi=144)
    im = ax.imshow(data, interpolation='nearest')
    fig.colorbar(im, orientation='vertical')
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(data)), labels=data.index, rotation=90)
    ax.set_yticks(np.arange(len(data)), labels=data.index)
    # Loop over data dimensions and create text annotations
    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            ax.text(j, i, round(data.to_numpy()[i, j], 2),
                    ha="center", va="center", color="black")
    # Setting the title of the heatmap
    if num == 1:
        ax.set_title("Correlation Coefficient Matrix of India", fontsize=12)
    else:
        ax.set_title("Correlation Coefficient Matrix of Australia",
                     fontsize=12)


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

# Removing the NaN vaues
# Since there is risk of getting the entire dataframe getting deleted while
# removing the NaN values, so removing the NaN values from the required info
df_india_co2 = pd.DataFrame(
    df_india['CO2 emissions from liquid fuel consumption (kt)'])
df_india_co2 = df_india_co2.dropna()
df_usa_co2 = pd.DataFrame(
    df_usa['CO2 emissions from liquid fuel consumption (kt)'])
df_usa_co2 = df_usa_co2.dropna()
df_somalia_co2 = pd.DataFrame(
    df_somalia['CO2 emissions from liquid fuel consumption (kt)'])
df_somalia_co2 = df_somalia_co2.dropna()
# Calculating the statistics of CO2 emissions from liquid fuel consumption (kt)
# of India from 1960 to 2021
min_co2 = df_india_co2['CO2 emissions from liquid fuel consumption (kt)'].min()
max_co2 = df_india_co2['CO2 emissions from liquid fuel consumption (kt)'].max()
mean_co2 = np.mean(df_india_co2
                   ['CO2 emissions from liquid fuel consumption (kt)'])
std_co2 = np.std(df_india_co2
                 ['CO2 emissions from liquid fuel consumption (kt)'])
skew = stats.skew(df_india_co2
                  ['CO2 emissions from liquid fuel consumption (kt)'])
kurt = stats.kurtosis(df_india_co2
                      ['CO2 emissions from liquid fuel consumption (kt)'])
print("Minimum of CO2 emissions from liquid fuel consumption (kt) of India = ",
      min_co2)
print("Maximum of CO2 emissions from liquid fuel consumption (kt) of India = ",
      max_co2)
print("Mean of CO2 emissions from liquid fuel consumption (kt) of India = ",
      mean_co2)
print("S.D. of CO2 emissions from liquid fuel consumption (kt) of India = ",
      std_co2)
print('Skewness of CO2 emissions from liquid fuel consumption(kt) of India = ',
      skew)
print('Kurtosis of CO2 emissions from liquid fuel consumption(kt) of India = ',
      kurt)
