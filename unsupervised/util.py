import numpy as np
import pandas as pd

def import_data():
    data1 = pd.read_csv('phishing.csv', sep=',')
    data2 = pd.read_csv('optdigits.csv', sep=',')

    X1 = data1.values[:, 0:-1]
    Y1 = data1.values[:, -1]
    X2 = data2.values[:, 0:-1]
    Y2 = data2.values[:, -1]

    return X1, Y1, X2, Y2

def import_data_pca():
    data1 = pd.read_csv('phishing_ica.csv', sep=',')
    data2 = pd.read_csv('optical_ica.csv', sep=',')

    X1 = data1.values[:, 0:-1]
    Y1 = data1.values[:, -1]
    X2 = data2.values[:, 0:-1]
    Y2 = data2.values[:, -1]

    return X1, Y1, X2, Y2