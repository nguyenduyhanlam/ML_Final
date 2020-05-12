# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:03:42 2020

@author: lam.nguyen
"""
# Import necessary modules and set options
import pandas as pd
import numpy as np
import itertools

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, LarsCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")

# Load data
data = pd.read_csv("prostate_data.txt", sep = "\t")
print(data.head())

# Train-test split
y_train = np.array(data[data.train == "T"]['lpsa'])
y_test = np.array(data[data.train == "F"]['lpsa'])
X_train = np.array(data[data.train == "T"].drop(['lpsa', 'train'], axis=1))
X_test = np.array(data[data.train == "F"].drop(['lpsa', 'train'], axis=1))