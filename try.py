import pandas as pd
import numpy as np
import itertools
from scipy import stats
import scipy

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
names = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
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
size_predictor = X_train.shape[1]

# Centering for the training data
meany_train = y_train.mean()
y_train_centered = np.subtract(y_train, meany_train)
X_train_centered = np.subtract(X_train, np.mean(X_train, axis=0))

normx = scipy.sqrt((X_train_centered**2).sum(axis=0))
X_train_scaled = np.divide(X_train_centered, normx)

matx = np.asmatrix(X_train_scaled)
vecy = np.asmatrix(y_train_centered)

"""Main loop for LAR

Reference:
Least Angle Regression, Efron et al., 2004, The Annals of Statistics
"""
# Initial data
signs = scipy.zeros(size_predictor)
betas = scipy.zeros(size_predictor)
indices_predictor = scipy.arange(size_predictor)
vecy_fitted = scipy.zeros_like(vecy)
beta_lars = [[0]*size_predictor]

for k in range(size_predictor):
    vecc = (vecy-vecy_fitted) @ matx
    vecc_abs = scipy.absolute(vecc)
    
    maxc = vecc_abs.max()
    mask_maxc = scipy.isclose(vecc_abs, maxc)
    indices_predictor = np.reshape(
        indices_predictor, mask_maxc.shape, order='C')
    active = indices_predictor[mask_maxc]
    signs = scipy.where(vecc.item(0,active[0]) > 0, 1, -1)

    matx_active = signs*matx[:, active]

    u, s, vh = scipy.linalg.svd(matx_active, full_matrices=False)
    matg = vh.T @ scipy.diag(s**2) @ vh
    matg_inv = vh.T @ scipy.diag(scipy.reciprocal(s**2)) @ vh
    vec1 = scipy.ones(len(active))
    scalara = (matg_inv.sum())**(-.5)

    vecw = scalara * matg_inv.sum(axis=1)
    vecw = np.reshape(vecw, (vecw.shape[0],1))
    vecu = matx_active @ vecw

    veca = matx.T @ vecu

    if k < size_predictor-1:
        inactive = indices_predictor[scipy.invert(mask_maxc)]
        arr_gamma = scipy.concatenate([(maxc - np.take(vecc, inactive)) / (scalara - np.take(veca, inactive)),
                                       (maxc + np.take(vecc, inactive)) / (scalara + np.take(veca, inactive))]).ravel()
        scalargamma = arr_gamma[arr_gamma > 0].min()
    else:
        scalargamma = maxc / scalara

    vecy_fitted += (scalargamma * vecu).T
    betas[active] += scalargamma * signs
    beta_lars.append(list(betas))

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(1, 1, 1)
l1length_ols_beta = sum(abs(coeff) for coeff in beta_lars[-1])
l1length_lars_beta = [sum(abs(coeff) for coeff in beta)/l1length_ols_beta for beta in beta_lars]
ax.plot(l1length_lars_beta, beta_lars, 'o-')
ax.grid()
ax.set_xlabel('Shrinkage Factor s')
ax.set_ylabel('Coefficients')
ax.legend(names)
plt.show()