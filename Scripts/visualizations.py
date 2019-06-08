'''
visualizations.py
'''

import sys
import math
import itertools as it
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, metrics, utils, base, preprocessing
from sklearn import neighbors, linear_model, tree, svm, ensemble, naive_bayes
from sklearn.model_selection import train_test_split
from sklearn.utils.fixes import signature
import model_specs as ms


def generate_histogram(dataframe, colname, color, binwidth, title):
    '''
    Generates a histogram for a given variable while EXCLUDING outlier values
    (both high and low) for that given variable. Rounds max and min values
    to the nearest (binwidth) to establish range.

    Inputs: dataframe (pandas dataframe)
      colname (str), the column to return a histogram for
      color (str), color for the histogram bars
      binwidth (int or float), number describing the width of histogram bins
      title (str), plot title

    Returns: matplotlib plot object (inline)
    '''
    outliers = identify_outliers(dataframe, colname)
    # Exclude outliers from the histogram
    df = dataframe[~outliers]
    maximum = binwidth * round(max(df[colname])/binwidth)
    minimum = binwidth * round(min(df[colname])/binwidth)
    bw = (maximum-minimum)/binwidth
    plt.hist(df[colname], color=color, edgecolor='black', bins=int(bw))
    plt.title(title)
    plt.xlabel(colname)
    plt.ylabel('Count')
    plt.show()


def generate_boxplot(dataframe, colname, category=None, hue=None):
    '''
    Generate a boxplot using pyplot to be printed inline

    Inputs: dataframe (pandas dataframe)
      colname (str): column of analysis
      category (str): by-group category
      hue (str): second by-group category
    '''
    if category:
        outliers = identify_outliers(dataframe, category)
        df = dataframe[~outliers]
        ax = sns.boxplot(x=colname, y=category, data=df, palette='Set1')
        title = "Box plot of " + colname + " by " + category
        plt.title(title)
        plt.show()

    elif hue:
        outliers = identify_outliers(dataframe, hue)
        df = dataframe[~outliers]
        ax = sns.boxplot(x=colname, hue=hue, data=df, palette='Set1')
        title = "Box plot of " + colname + " by " + hue
        plt.title(title)
        plt.show()

    elif hue and category:
        outliers = identify_outliers(dataframe, category)
        df = dataframe[~outliers]
        outliers = identify_outliers(dataframe, hue)
        df = dataframe[~outliers]
        ax = sns.boxplot(x=colname, y=category, hue=hue, data=df, palette='Set1')
        title = "Box plot of " + colname + " by " + " and ".join([category, hue])
        plt.title(title)
        plt.show()

    else:
        ax = sns.boxplot(x=colname, data=dataframe, palette='Set1')
        title = "Box plot of " + colname
        plt.title(title)
        plt.show()


def correlation_heatmap(dataframe, size=10):
    '''
    Create a correlation heatmap from a dataframe's numeric variables.
    Inspiration from https://stackoverflow.com/questions/39409866/correlation-heatmap

    Inputs: dataframe (pandas dataframe)
      size (int): plot size to be displayed
    '''
    corr = dataframe.corr()
    sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
    plt.title("Correlation Matrix Heatmap")
    plt.show()


def plot_scatter(dataframe, x, y, outliers=True):
    '''
    Create a scatter plot using matplotlib.pyplot.

    Inputs: dataframe (pandas df)
      x (str): horizontal-axis var, column from dataframe
      y (str): vertical-axis var, column from dataframe

    Returns: nothing, prints graphic inline
    '''
    df = dataframe
    if not outliers:
        outlier_x = identify_outliers(dataframe, x)
        outlier_y = identify_outliers(dataframe, y)
        df = dataframe[~(outlier_x|outlier_y)]
    ax = plt.scatter(df[x], df[y], s=1, c="black")
    title = "Plot of " + x + " against " + y
    plt.title(title)
    plt.show()