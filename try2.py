# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:01:28 2020

@author: User
"""

import scipy
import scipy.stats
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("prostate_data.txt", sep = "\t",index_col=0)
dfy = df.pop('lpsa')
dfmask = df.pop('train')
size_predictor = df.columns.size

dfmask_train = dfmask == 'T'
dfx_train = df[dfmask_train]
dfy_train = dfy[dfmask_train]
size_train = dfmask_train.sum()

dfmask_test = dfmask == 'F'
dfx_test = df[dfmask_train]
dfy_test = dfy[dfmask_train]
size_test = dfmask_train.sum()

# Centering for the training data
meany_train = dfy_train.mean()
dfy_train_centered = dfy_train.subtract(meany_train)
dfx_train_centered = dfx_train.subtract(dfx_train.mean())

normx = scipy.sqrt((dfx_train_centered**2).sum())
dfx_train_scaled = dfx_train_centered.divide(normx)

matx = dfx_train_scaled.values
vecy = dfy_train_centered.values
y = df['lpsa']

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
    vecc = matx.T @ (vecy-vecy_fitted)
    vecc_abs = scipy.absolute(vecc)
    
    maxc = vecc_abs.max()
    mask_maxc = scipy.isclose(vecc_abs, maxc)
    active = indices_predictor[mask_maxc]
    signs = scipy.where(vecc[active] > 0, 1, -1)

    matx_active = signs*matx[:, active]

    u, s, vh = scipy.linalg.svd(matx_active, full_matrices=False)
    matg = vh.T @ scipy.diag(s**2) @ vh
    matg_inv = vh.T @ scipy.diag(scipy.reciprocal(s**2)) @ vh
    vec1 = scipy.ones(len(active))
    scalara = (matg_inv.sum())**(-.5)

    vecw = scalara * matg_inv.sum(axis=1)
    vecu = matx_active @ vecw

    veca = matx.T @ vecu

    if k < size_predictor-1:
        inactive = indices_predictor[scipy.invert(mask_maxc)]
        arr_gamma = scipy.concatenate([(maxc-vecc[inactive])/(scalara-veca[inactive]),
                                       (maxc+vecc[inactive])/(scalara+veca[inactive])])
        scalargamma = arr_gamma[arr_gamma > 0].min()
    else:
        scalargamma = maxc/scalara

    vecy_fitted += scalargamma*vecu
    betas[active] += scalargamma*signs
    beta_lars.append(list(betas))
    
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
l1length_ols_beta = sum(abs(coeff) for coeff in beta_lars[-1])
l1length_lars_beta = [sum(abs(coeff) for coeff in beta)/l1length_ols_beta for beta in beta_lars]
ax.plot(l1length_lars_beta, beta_lars, 'o-')
ax.grid()
ax.set_xlabel('Shrinkage Factor s')
ax.set_ylabel('Coefficients')
ax.legend(dfx_train.columns)
plt.show()