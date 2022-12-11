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
    im = ax.imshow(data, interpolation='nearest', cmap="Spectral")
    fig.colorbar(im, orientation='vertical')
    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(data)), labels=data.index, rotation=90,
                  fontsize=12)
    ax.set_yticks(np.arange(len(data)), labels=data.index, fontsize=12)
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
# -----------------------------------------------------------------------------
# scale to millions (for better layout)
df_india_co2["CO2 in billions"] = df_india_co2[
    "CO2 emissions from liquid fuel consumption (kt)"] / 1.0e6
df_usa_co2["CO2 in billions"] = df_usa_co2[
    "CO2 emissions from liquid fuel consumption (kt)"] / 1.0e6
df_somalia_co2["CO2 in billions"] = df_somalia_co2[
    "CO2 emissions from liquid fuel consumption (kt)"] / 1.0e6

# Plotting the CO2 emmissions of India, USA and Somalia
# Setting the figure size and resolution
plt.figure(figsize=(15, 8), dpi=144)
# Plot of India
plt.plot(df_india_co2.index, df_india_co2["CO2 in billions"], 'r--',
         label='India')
# Plot of USA
plt.plot(df_usa_co2.index, df_usa_co2["CO2 in billions"], 'g-.',
         label='USA')
# Plot of Somalia
plt.plot(df_somalia_co2.index, df_somalia_co2["CO2 in billions"],
         label='Somalia')
# Setting the tittle of the plot
plt.title("CO2 from liquid fuel consumption of India, USA & Somalia(10^9 t)",
          fontsize=15)
# Setting the labels of x and y axes
plt.xlabel('Year', fontsize=12)
plt.ylabel('CO2 emissions from liquid fuel consumption (10^9 t)',
           fontsize=12)
plt.legend()
# Rotating the x labels by 90 Deg
plt.xticks(rotation=90)
# plt.savefig('CO2_plot.jpg')
plt.show()

# -----------------------------------------------------------------------------
# Creating new dataframes for the 3 countries having Cereal Yeild as column
df_india_cereal = pd.DataFrame(df_india['Cereal yield (kg per hectare)'])
df_usa_cereal = pd.DataFrame(df_usa['Cereal yield (kg per hectare)'])
df_somalia_cereal = pd.DataFrame(df_somalia['Cereal yield (kg per hectare)'])
# Creating a new dataframe having years of 1995, 2000, 2005, 2010, 2015 and
# 2020
df_cereal = pd.DataFrame().assign(
    India=df_india_cereal.loc[
        ['1995', '2000', '2005', '2010', '2015', '2020']],
    USA=df_usa_cereal.loc[
        ['1995', '2000', '2005', '2010', '2015', '2020']],
    Somalia=df_somalia_cereal.loc[
        ['1995', '2000', '2005', '2010', '2015', '2020']])

# Barplot showing Cereal yeild of India, USA, and Somalia
x_axis = np.arange(6)
width = 0.25
# Setting the figure size and resolution
plt.figure(figsize=(15, 8), dpi=144)
# Plotting the bar of Somalia
plt.bar(x_axis, df_cereal['Somalia'], width, label='Somalia',
        edgecolor='black')
# Plotting the bar of India
plt.bar(x_axis+width, df_cereal['India'], width, label='India',
        edgecolor='black')
# Plotting the bar of USA
plt.bar(x_axis+width*2, df_cereal['USA'], width, label='USA',
        edgecolor='black')
# Setting the x-ticks
plt.xticks(x_axis+width, ['1995', '2000', '2005', '2010', '2015', '2020'])
# Setting the x and y labels
plt.xlabel('Year', fontsize=12)
plt.ylabel('Cereal yield (kg per hectare)', fontsize=12)
# Setting the title of the plot
plt.title('Cereal yield (kg per hectare) of Somalia, India & USA', fontsize=15)
plt.legend()
plt.show()

# -----------------------------------------------------------------------------
# Creating dataframes of Correlation Coefficient
# Using groupby and get_group function to retrieve information of India
# from the original dataframe
grp_india = df.groupby('Country Name').get_group('India')
# Extracting the required Indicator from the dataframe
grp_india = grp_india.loc[
    (grp_india["Indicator Name"] == 'Population, total') |
    (grp_india["Indicator Name"] == "Forest area (sq. km)") |
    (grp_india["Indicator Name"] ==
     "CO2 emissions from solid fuel consumption (kt)") |
    (grp_india["Indicator Name"] ==
     "Electric power consumption (kWh per capita)") |
    (grp_india["Indicator Name"] == "Agricultural land (sq. km)") |
    (grp_india["Indicator Name"] ==
     "Agriculture, forestry, and fishing, value added (% of GDP)")]
# Transposing the dataframe to get years as the indices
grp_india = grp_india.transpose()
# Removing the unwanted rows for the calculation of Correlation Coefficient
grp_india.drop(['Country Name', 'Country Code', 'Indicator Code'],
               inplace=True)
# Setting the Indicator names as the columns of the dataframe
grp_india.columns = grp_india.iloc[0]
# Removing the NaN values from the dataframe
grp_india.dropna(inplace=True)
# Removing the Indicator name row from the dataframe
grp_india.drop('Indicator Name', inplace=True)
# Changing the datatype of the contents of the dataframe to float for the
# purpose of calculation of correlation coefficient
grp_india = grp_india.astype(float)
# Calculating correlation coefficient of the data frame
india_corr = grp_india.corr()

# Using groupby and get_group function to retrieve information of Australia
# from the original dataframe
grp_aus = df.groupby('Country Name').get_group('Australia')
# Extracting the required Indicator from the dataframe
grp_aus = grp_aus.loc[
    (grp_aus["Indicator Name"] == 'Population, total') |
    (grp_aus["Indicator Name"] == "Forest area (sq. km)") |
    (grp_aus["Indicator Name"] ==
     "CO2 emissions from solid fuel consumption (kt)") |
    (grp_aus["Indicator Name"] ==
     "Electric power consumption (kWh per capita)") |
    (grp_aus["Indicator Name"] == "Agricultural land (sq. km)") |
    (grp_aus["Indicator Name"] ==
     "Agriculture, forestry, and fishing, value added (% of GDP)")]
# Transposing the dataframe to get years as the indices
grp_aus = grp_aus.transpose()
# Removing the unwanted rows for the calculation of Correlation Coefficient
grp_aus.drop(['Country Name', 'Country Code', 'Indicator Code'], inplace=True)
# Setting the Indicator names as the columns of the dataframe
grp_aus.columns = grp_aus.iloc[0]
# Removing the NaN values from the dataframe
grp_aus.dropna(inplace=True)
# Removing the Indicator name row from the dataframe
grp_aus.drop('Indicator Name', inplace=True)
# Changing the datatype of the contents of the dataframe to float for the
# purpose of calculation of correlation coefficient
grp_aus = grp_aus.astype(float)
aus_corr = grp_aus.corr()

# Functoin call to plot heatmap of Correlation coefficient of India and
# Australia
heatmap(india_corr, 1)
heatmap(aus_corr, 2)

# -----------------------------------------------------------------------------
# To plot correlation coeficient between Population, total and Agricultural
# land over the period from 1990 to 2019 during interval of 10 years of
# Australia
# Filtering out Australia from the original dataframe
grp_corr = df.groupby('Country Name').get_group('Australia')
# Retrieving required Indicators from the dataframe
grp_corr = grp_corr.loc[
    (grp_corr["Indicator Name"] == 'Population, total') |
    (grp_corr["Indicator Name"] == "Agricultural land (sq. km)")]
# # Transposing the dataframe to get years as the indices
grp_corr = grp_corr.transpose()
# Removing the unwanted rows for the calculation of Correlation Coefficient
grp_corr.drop(['Country Name', 'Country Code', 'Indicator Code'], inplace=True)
# Setting the Indicator names as the columns of the dataframe
grp_corr.columns = grp_corr.iloc[0]
# Removing the NaN values from the dataframe
grp_corr.dropna(inplace=True)
# Removing the Indicator name row from the dataframe
grp_corr.drop('Indicator Name', inplace=True)
# Changing the datatype of the contents of the dataframe to float for the
# purpose of calculation of correlation coefficient
grp_corr = grp_corr.astype(float)
# Renaming the columns of the dataframe
grp_corr = grp_corr.rename(columns={'Population, total': 'Population',
                                    "Agricultural land (sq. km)": "Agri land"})
# Creating an empty list to store the correlation coefficient
corr = []
# Setting the range for 'for loop'
range = [0, 10, 20, 30, 40, 50]
# For loop to append correlation coefficients into the list
for i in range:
    corr.append(grp_corr.loc[
        grp_corr.index[i]:grp_corr.index[i+9]]['Population'].corr(grp_corr.loc[
            grp_corr.index[i]:grp_corr.index[i+9]]['Agri land']))
    i = i+9
# Rounding the correlation coefficients to the nearest 2 decimal points
corr = [round(num, 2) for num in corr]
# Creating a list for the x labels of the plot of correlation coefficient over
# years
time = ['1960-1969', '1970-1979', '1980-1989',
        '1990-1999', '2000-2009', '2010-2019']

# To plot correlation coeficient over the period from 1990 to 2019 during
# interval of 10 years of Australia
plt.figure(figsize=(12, 8), dpi=144)
plt.plot(time, corr, label='Australia')
# For lopp to annotate the correlation coefficient values on the graph
for i, j in zip(time, corr):
    plt.annotate(str(j), xy=(i, j-0.02), fontsize=12)
# Setting the x and y labels
plt.xlabel("Year", fontsize=12)
plt.ylabel('Correlation Coefficient', fontsize=12)
# Setting the grapg title
plt.title(
    'Correlation  Coefficient between Population and '
    'Agricultural Land of Australia', fontsize=15)
plt.legend()
plt.show()
