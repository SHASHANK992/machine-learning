from util import *
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis

phishing_X, phishing_Y, optdigits_X, optdigits_Y = import_data()


# Optical Digits
dims = range(1,64)
kurt = {}
for dim in dims:
    ica = FastICA(n_components=dim)
    tmp = ica.fit_transform(optdigits_X)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt)
kurt.to_csv('optdigits_kurt.csv')

# Phishing Dataset
dims = range(1,30)
kurt = {}
for dim in dims:
    ica = FastICA(n_components=dim)
    tmp = ica.fit_transform(phishing_X)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt)
kurt.to_csv('phishing_kurt.csv')





