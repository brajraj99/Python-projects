#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 21:58:25 2023

@author: risingphoenix
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing as prep
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import err_ranges as err
import scipy.optimize as opt


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


def scatter_plot(data1, data2, a, b):
    """
    Function takes in two dataframes and two integer values of the years as
    arguments and produce a plot with 2 subplots of scatter plots.

    Parameters
    ----------
    data1 : TYPE: Dataframe
        Dataframe of the CO2 emmissions and Cereal Yield for the first year.
    data2 : TYPE: Dataframe
        Dataframe of the CO2 emmissions and Cereal Yield for the second year.
    a : TYPE: Integer
        1st Year
    b : TYPE: Integer
        2nd Year

    Yields
    ------
    None.

    """
    # Setting the figure size and axes of the subplot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=200)
    # Title of the whole plot
    fig.suptitle(
        'Scatter Plots of CO2 emmissions vs Cereal Yield for 1990 and 2019',
        fontsize=18)
    # Plotting the data
    ax1.scatter(data1['co2'], data1['cereal'], color='red')
    ax2.scatter(data2['co2'], data2['cereal'], color='green')
    # Setting the x and y labels and title of each subplots
    ax1.set_xlabel('CO2 emissions (kg per 2015 US$ of GDP)', fontsize=12)
    ax2.set_xlabel('CO2 emissions (kg per 2015 US$ of GDP)', fontsize=12)
    ax1.set_ylabel('Cereal yield (kg per hectare)', fontsize=12)
    ax1.set_ylabel('Cereal yield (kg per hectare)', fontsize=12)
    ax1.set_title('CO2 emmissions vs Cereal Yield for %d' % a, fontsize=15)
    ax2.set_title('CO2 emmissions vs Cereal Yield for %d' % b, fontsize=15)
    plt.show()


def clustering(data, ncluster, year):
    """
    The function takes number of clusters and the dataset as a dataframe and do
    clustering analysis on the dataset. Finally it plots the data showing the
    clusters as scatter plot

    Parameters
    ----------
    data : TYPE: Dataframe
        Dataset as a dataframe for clustering.
    ncluster : TYPE: Integer
        Number of clusters
    year : TYPE: Integer
        Year to be printed on the graph.

    Yields
    ------
    clusters1 : TYPE: Dataframe
        Dataframe containing Countries and their cluster numbers.

    """
    global kmeans
    # set up the clusterer for number of clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(data)
    # labels is the number of the associated clusters of (x,y) points
    labels = kmeans.labels_
    clusters1 = pd.DataFrame(labels, index=data.index, columns=['Cluster ID'])
    # extract the estimated cluster centres
    centers = kmeans.cluster_centers_
    # calculate the silhoutte score
    sil = skmet.silhouette_score(data, labels)
    print(sil)

    # plot using the labels to select colour
    plt.figure(figsize=(15, 15), dpi=200)
    col = ["blue", "orange", "green", "tab:red", "tab:purple",
           "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    # loop over the different labels
    for l in range(ncluster):
        plt.plot(data[labels == l]["co2"], data[labels == l]["cereal"], "o",
                 markersize=10, color=col[l], alpha=0.5, label=("Cluster", l+1))
    # show cluster centres
    for ic in range(ncluster):
        xc, yc = centers[ic, :]
        # plotting the cluster centre on the same plot
        plt.plot(xc, yc, "dk", markersize=10)
    plt.xlabel("CO2 emissions (kg per 2015 US$ of GDP)", fontsize=12)
    plt.ylabel("Cereal yield (kg per hectare)", fontsize=12)
    plt.title("Scatter plot showing the Clustering of different countries for the year %d" % year, fontsize=15)
    plt.legend()
    plt.show()
    
    return clusters1


def prediction(testdata, year):
    """
    The function takes in a dataset as dataframe whose clustering has to be
    predicted. The dataset with the predicted clustering will be plotted as
    a scatter plot.

    Parameters
    ----------
    testdata : TYPE: Dataframe
        Dataset as a dataframe for clustering to be predicted.
    year : TYPE: Integer
        Year of the dataset.

    Yields
    ------
    clusters2 : TYPE: Dataframe
        Dataframe containing Countries and their cluster numbers.

    """
    labels = kmeans.predict(testdata)
    clusters2 = pd.DataFrame(labels, index=testdata.index, columns=
                             ['Cluster ID'])
    # extract the estimated cluster centres
    centers = kmeans.cluster_centers_
    # calculate the silhoutte score
    sil = skmet.silhouette_score(testdata, labels)
    print(sil)
    ncluster = list(np.unique(labels))

    # plot using the labels to select colour
    plt.figure(figsize=(15, 15), dpi=144)
    col = ["blue", "orange", "green", "tab:red", "tab:purple",
           "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    # loop over the different labels
    for l in range(len(ncluster)):
        plt.plot(testdata[labels == l]["co2"], testdata[labels == l]["cereal"],
                 "o", markersize=10, color=col[l], alpha=0.5, label=
                 ("Cluster", l+1))
    # show cluster centres
    for ic in range(len(ncluster)):
        xc, yc = centers[ic, :]
        # plotting the cluster centre on the same plot
        plt.plot(xc, yc, "dk", markersize=10)
    plt.xlabel("CO2 emissions (kg per 2015 US$ of GDP)", fontsize=12)
    plt.ylabel("Cereal yield (kg per hectare)", fontsize=12)
    plt.title("Scatter plot showing the PREDICTED Clustering of different countries for the year %d" % year,fontsize=15)
    plt.legend()
    plt.show()

    return clusters2


def best_cluster(data, year):
    """
    Function takes in a dataframe as argument and do clustering on that dataset
    SSE and Silhoutte score will also be calculated for each number of clusters
    and plotted against the number of clusters to find the optimum number of
    clusters.

    Parameters
    ----------
    data : Dataframe
        Dataframe of the CO2 emmissions and Cereal Yield for an year.
    n : TYPE: Integer
        Year

    Returns
    -------
    None.

    """
    # Creating an empty array for SSE and silhouette_coefficients
    SSE = []
    silhouette_coefficients = []

    # Calculating the SSE for each number of clusters
    for k in range(2, 11):
        kmeans = cluster.KMeans(n_clusters=k)
        kmeans.fit(data)
        labels = kmeans.labels_
        SSE.append(kmeans.inertia_)
        score = skmet.silhouette_score(data, kmeans.labels_)
        silhouette_coefficients.append(score)

    # Plotting the Number of Clusters against SSE
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=144)
    # Title of the whole plot
    fig.suptitle('SSE and Silhouette Score to find Optimum Number of Clusters for the year %d' % year, fontsize=18)
    ax1.plot(range(2, 11), SSE)
    ax1.set_xticks(range(2, 11))
    ax1.set_xlabel('Number of Clusters', fontsize=12)
    ax1.set_ylabel('Sum of Squared Error, SSE', fontsize=12)
    ax1.set_title('SSE v/s Number of Clusters', fontsize=15)
    ax2.plot(range(2, 11), silhouette_coefficients)
    ax2.set_xticks(range(2, 11))
    ax2.set_title('Silhouette Coefficient v/s Number of Clusters', fontsize=15)
    ax2.set_xlabel('Number of Clusters', fontsize=12)
    ax2.set_ylabel('Silhouette Coefficient', fontsize=12)
    plt.show()
    # To find the index value of the maximum value of Silhouette Coefficient
    index_max = max(range(len(silhouette_coefficients)),
                    key=silhouette_coefficients.__getitem__)
    print("Maximum value of Silhouette Coefficient = %f "
          % max(silhouette_coefficients))
    print("Optimum number of clusters = ", index_max+2)


def objective(x, a, b, c, d):
    """Cubic polynominal for the fitting"""

    f = a * x**3 + b * x**2 + c * x + d

    return f


def logistics(t, scale, growth, t0):
    """
    Computes logistics function with scale, growth rate and time of the
    turning point as free parameters
    """

    f = scale / (1.0 + np.exp(-growth * (t - t0)))

    return f


# -----------------------------------------------------------------------------
# Calling the function dataset() to read the csv data file and obtaining the
# original data frame and transposed dataframe
df, df_tran = dataset('API_19_DS2_en_csv_v2_4700503')

# Preprocessing the data
df_co2 = df.groupby('Indicator Name').get_group(
    'CO2 emissions (kg per 2015 US$ of GDP)')
df_cereal = df.groupby('Indicator Name').get_group(
    'Cereal yield (kg per hectare)')
df_co2.reset_index(inplace=True)
# Dropping the unwanted columns and blank cells
df_co2 = df_co2.drop(
 ['index', 'Country Code', 'Indicator Name', 'Indicator Code', '1960', '2021'],
 axis=1)
df_cereal.reset_index(inplace=True)
# Dropping the unwanted columns and blank cells
df_cereal = df_cereal.drop(
 ['index', 'Country Code', 'Indicator Name', 'Indicator Code', '1960', '2021'],
 axis=1)
# Setting the index of the dataframes
df_co2 = df_co2.set_index('Country Name')
df_cereal = df_cereal.set_index('Country Name')
# Slicing the required data from the dataframe
df_co2 = df_co2.loc[:, '1990':'2019']
df_cereal = df_cereal.loc[:, '1990':'2019']
# Replacing the nan values with 0
df_co2.fillna(0, inplace=True)
df_cereal.fillna(0, inplace=True)
# Removing the outliers from the data, which adversely affects the clustering
df_co2.drop(labels=['St. Vincent and the Grenadines', 'United Arab Emirates',
                    'Kuwait', 'Oman'], axis=0, inplace=True)
df_cereal.drop(labels=['Mongolia',
                       'Syrian Arab Republic'], axis=0, inplace=True)
# Creating new dataframe having both co2 and cereal data
df_1990 = pd.DataFrame().assign(co2=df_co2['1990'], cereal=df_cereal['1990'])
df_2019 = pd.DataFrame().assign(co2=df_co2['2019'], cereal=df_cereal['2019'])
# Removing the zero values from the dataframe
df_1990.replace(0, np.nan, inplace=True)
df_1990.dropna(inplace=True)
df_2019.replace(0, np.nan, inplace=True)
df_2019.dropna(inplace=True)
# Normalizing the values of the dataframe for the purpuse of clustering
dfn_1990 = pd.DataFrame(prep.normalize(df_1990, axis=0),
                        index=df_1990.index, columns=df_1990.columns)
dfn_2019 = pd.DataFrame(prep.normalize(df_2019, axis=0),
                        index=df_2019.index, columns=df_2019.columns)
# Calling each function for scatter plot, finding the optimum number of cluster
# clustering and prediction
scatter_plot(dfn_1990, dfn_2019, 1990, 2019)
best_cluster(dfn_1990, 1990)
a_1990 = clustering(dfn_1990, 3, 1990)
b_2019 = prediction(dfn_2019, 2019)

# -----------------------------------------------------------------------------
# Preprocessing the dataset for curve fitting
# changing the header of the transposed datafram and populating it with the
# country names
header1 = df_tran.iloc[0]
df_tran = df_tran[1:]
df_tran.columns = header1

# Extracting information from the transposed matrix for different countries
# Creating a dataframe for China and changing the column names of that
# dataframe as the Indicator name
df_china = df_tran['China']
# The header extracted below can be used for headers of different countries
header_country = df_china.iloc[1]
# Renaming the columns as Indicator names
df_china.columns = header_country
# Dropping the unwanted rows from the dataframe
df_china = df_china.drop(['Indicator Name', 'Country Code', 'Indicator Code'])
data = pd.DataFrame(df_china['Population, total'])
data.reset_index(inplace=True)
data.rename(columns={"index": "year"}, inplace=True)
data = data.astype(float)

# choose the input and output variables
x, y = data['year'], data['Population, total']

# estimated turning year: 1965
# population in 1965: about 800 million
# increase scale factor and growth rate until rough fit
popt = [0.7e9, 0.05, 1965]
data["pop_log"] = logistics(data['year'], *popt)

plt.figure()
plt.title("logistics function")
plt.plot(x, y, 'r+', label="data")
plt.plot(x, data["pop_log"], label="fit")
plt.legend()
plt.xlabel("year")
plt.ylabel("population")
plt.show()

# Aplying the curve fit
popt, covar = opt.curve_fit(logistics, x, y, p0=(0.7e9, 0.05, 1967))
print("Fit parameter", popt)
data["pop_log"] = logistics(data['year'], *popt)

# Plotting the curve fitted line on the data
plt.figure()
plt.title("logistics function")
plt.plot(x, y, 'r+', label="data")
plt.plot(x, data["pop_log"], label="fit")
plt.legend()
plt.xlabel("year")
plt.ylabel("population")
plt.show()

# Extracting the sigmas from the diagonal of the covariance matrix
sigma = np.sqrt(np.diag(covar))
print(sigma)

# Predicting the population of 2030 from the curve fit
print("\nForcasted population:")
low, high = err.err_ranges(2030, logistics, popt, sigma)
mean1 = (high+low) / 2.0
pm1 = (high-low) / 2.0
print('2030: %.3f +/- %.3f' % (mean1, pm1))
# Predicting the population of 2040 from the curve fit
low, high = err.err_ranges(2040, logistics, popt, sigma)
mean2 = (high+low) / 2.0
pm2 = (high-low) / 2.0
print('2040: %.3f +/- %.3f' % (mean2, pm2))
# Predicting the population of 2050 from the curve fit
low, high = err.err_ranges(2050, logistics, popt, sigma)
mean3 = (high+low) / 2.0
pm3 = (high-low) / 2.0
print('2050: %.3f +/- %.3f' % (mean3, pm3))

# Calculating the confidence range to plot on the plot
low, high = err.err_ranges(x, logistics, popt, sigma)

# Plotting data, fitted line, confidence range and predicted population
plt.figure(figsize=(10, 8), dpi=200)
plt.title("Curve Fit for the Population of China over Time", fontsize=15)
plt.plot(x, y, 'r+', label="Plotted Data")
plt.plot(x, data["pop_log"], color='green', label="Fitted Curve")
plt.plot(2030, mean1, 'r+', markersize=10)
plt.annotate('Pop @ 2030 = \n%.3f' % mean1, xy=(2030, 1.4e9),
             xytext=(2022, 1.37e9), fontsize=12)
plt.fill_between(x, low, high, color='yellow', alpha=0.7)
plt.legend(fontsize=12)
plt.xlabel("Time", fontsize=12)
plt.ylabel("Population, total", fontsize=12)
plt.show()
