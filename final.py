# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:03:42 2020

@author: lam.nguyen
"""
# Import necessary modules and set options
import pandas as pd
import numpy as np
import itertools
from scipy import stats

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV, LarsCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import warnings
import seaborn as sn
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# Load data
data = pd.read_csv("prostate_data.txt", sep = "\t")
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
print("Some data samples:")
print(data.head())

dataStat = data.describe()
print("Data statistical detail:")
print(dataStat)

correlationMatrix = data.corr()
print("Correlation matrix:")
print(correlationMatrix)

sn.heatmap(correlationMatrix, annot=True)
plt.show()

# Train-test split
y_train = np.array(data[data.train == "T"]['lpsa'])
y_test = np.array(data[data.train == "F"]['lpsa'])
X_train = np.array(data[data.train == "T"].drop(['lpsa', 'train'], axis=1))
X_test = np.array(data[data.train == "F"].drop(['lpsa', 'train'], axis=1))

##############################################################################
########################## LINEAR REGRESSION #################################
##############################################################################
print("##############################################################################")
print("Linear Regression")
linreg_model = LinearRegression(normalize=True).fit(X_train, y_train)
linreg_prediction = linreg_model.predict(X_test)
linreg_mae = mean_squared_error(y_test, linreg_prediction)
linreg_coefs = dict(
    zip(['Intercept'] + data.columns.tolist()[:-1], 
        np.round(np.concatenate((linreg_model.intercept_, linreg_model.coef_), 
        axis=None), 3))
)

print('Linear Regression MSE: {}'.format(np.round(linreg_mae, 3)))
print('Linear Regression coefficients:', linreg_coefs)

##############################################################################
####################### BEST SUBSET REGRESSION ###############################
##############################################################################
print("##############################################################################")
print("Best Subset Regression")
results = pd.DataFrame(columns=['num_features', 'features', 'MSE'])

# Loop over all possible numbers of features to be included
for k in range(1, X_train.shape[1] + 1):
    # Loop over all possible subsets of size k
    for subset in itertools.combinations(range(X_train.shape[1]), k):
        subset = list(subset)
        linreg_model = LinearRegression(normalize=True).fit(X_train[:, subset], y_train)
        linreg_prediction = linreg_model.predict(X_test[:, subset])
        linreg_mae = mean_squared_error(y_test, linreg_prediction)
        results = results.append(pd.DataFrame([{'num_features': k,
                                                'features': subset,
                                                'MSE': linreg_mae}]))

# Inspect best combinations
results = results.sort_values('MSE').reset_index()
print(results.head())

# Fit best model
best_subset_model = LinearRegression(normalize=True).fit(X_train[:, results['features'][0]], y_train)
best_subset_coefs = dict(
    zip(['Intercept'] + data.columns.tolist()[:-1], 
        np.round(np.concatenate((best_subset_model.intercept_, best_subset_model.coef_), axis=None), 3))
)

print('Best Subset Regression MSE: {}'.format(np.round(results['MSE'][0], 3)))
print('Best Subset Regression coefficients:', best_subset_coefs)

##############################################################################
########################## RIDGE REGRESSION ##################################
##############################################################################
print("##############################################################################")
print("Ridge Regression")
ridge_cv = RidgeCV(normalize=True, alphas=np.logspace(-10, 5, 1000), gcv_mode='svd')
ridge_model = ridge_cv.fit(X_train, y_train)
ridge_prediction = ridge_model.predict(X_test)
ridge_mae = mean_squared_error(y_test, ridge_prediction)
ridge_coefs = dict(
    zip(['Intercept'] + data.columns.tolist()[:-1], 
        np.round(np.concatenate((ridge_model.intercept_, ridge_model.coef_), 
                                axis=None), 3))
)

print('Ridge Regression MSE: {}'.format(np.round(ridge_mae, 3)))
print('Ridge Regression coefficients:', ridge_coefs)

##############################################################################
############################### LASSO ########################################
##############################################################################
print("##############################################################################")
print("LASSO")
lasso_cv = LassoCV(normalize=True, alphas=np.logspace(-10, 2, 1000))
lasso_model = lasso_cv.fit(X_train, y_train)
lasso_prediction = lasso_model.predict(X_test)
lasso_mae = mean_squared_error(y_test, lasso_prediction)
lasso_coefs = dict(
    zip(['Intercept'] + data.columns.tolist()[:-1], 
        np.round(np.concatenate((lasso_model.intercept_, lasso_model.coef_), axis=None), 3))
)

print('LASSO MSE: {}'.format(np.round(lasso_mae, 3)))
print('LASSO coefficients:', lasso_coefs)

##############################################################################
############################ ELASTIC NET #####################################
##############################################################################
print("##############################################################################")
print("ELASTIC NET")
elastic_net_cv = ElasticNetCV(normalize=True, alphas=np.logspace(-10, 2, 1000), 
                              l1_ratio=np.linspace(0, 1, 100))
elastic_net_model = elastic_net_cv.fit(X_train, y_train)
elastic_net_prediction = elastic_net_model.predict(X_test)
elastic_net_mae = mean_squared_error(y_test, elastic_net_prediction)
elastic_net_coefs = dict(
    zip(['Intercept'] + data.columns.tolist()[:-1], 
        np.round(np.concatenate((elastic_net_model.intercept_, 
                                 elastic_net_model.coef_), axis=None), 3))
)

print('Elastic Net MSE: {}'.format(np.round(elastic_net_mae, 3)))
print('Elastic Net coefficients:', elastic_net_coefs)

##############################################################################
###################### LEAST ANGLE REGRESSION ################################
##############################################################################
print("##############################################################################")
print("LEAST ANGLE REGRESSION")
LAR_cv = LarsCV(normalize=True)
LAR_model = LAR_cv.fit(X_train, y_train)
LAR_prediction = LAR_model.predict(X_test)
LAR_mae = mean_squared_error(y_test, LAR_prediction)
LAR_coefs = dict(
    zip(['Intercept'] + data.columns.tolist()[:-1], 
        np.round(np.concatenate((LAR_model.intercept_, LAR_model.coef_), axis=None), 3))
)

print('Least Angle Regression MSE: {}'.format(np.round(LAR_mae, 3)))
print('Least Angle Regression coefficients:', LAR_coefs)

##############################################################################
################## PRINCIPAL COMPONENTS REGRESSION ###########################
##############################################################################
print("##############################################################################")
print("PRINCIPAL COMPONENTS REGRESSION")
regression_model = LinearRegression(normalize=True)
pca_model = PCA()
pipe = Pipeline(steps=[('pca', pca_model), ('least_squares', regression_model)])
param_grid = {'pca__n_components': range(1, 9)}
search = GridSearchCV(pipe, param_grid)
pcareg_model = search.fit(X_train, y_train)
pcareg_prediction = pcareg_model.predict(X_test)
pcareg_mae = mean_squared_error(y_test, pcareg_prediction)
n_comp = list(pcareg_model.best_params_.values())[0]
pcareg_coefs = dict(
   zip(['Intercept'] + ['PCA_comp_' + str(x) for x in range(1, n_comp + 1)], 
       np.round(np.concatenate((pcareg_model.best_estimator_.steps[1][1].intercept_, 
                                pcareg_model.best_estimator_.steps[1][1].coef_), axis=None), 3))
)

print('Principal Components Regression MSE: {}'.format(np.round(pcareg_mae, 3)))
print('Principal Components Regression coefficients:', pcareg_coefs)

##############################################################################
################## PARTIAL LEAST SQUARES REGRESSION ##########################
##############################################################################
print("##############################################################################")
print("PARTIAL LEAST SQUARES REGRESSION")
pls_model_setup = PLSRegression(scale=True)
param_grid = {'n_components': range(1, 9)}
search = GridSearchCV(pls_model_setup, param_grid)
pls_model = search.fit(X_train, y_train)
pls_prediction = pls_model.predict(X_test)
pls_mae = mean_squared_error(y_test, pls_prediction)
pls_coefs = dict(
  zip(data.columns.tolist()[:-1], 
      np.round(np.concatenate((pls_model.best_estimator_.coef_), axis=None), 3))
)

print('Partial Least Squares Regression MSE: {}'.format(np.round(pls_mae, 3)))
print('Partial Least Squares Regression coefficients:', pls_coefs)