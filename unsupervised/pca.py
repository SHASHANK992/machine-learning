from util import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

phishing_X, phishing_Y, optdigits_X, optdigits_Y = import_data()

pca = PCA(0.9)
pca.fit(phishing_X)
# print(pca.components_)
# print(pca.explained_variance_)

X_pca = pca.transform(phishing_X)
print("original shape:   ", phishing_X.shape)
print("transformed shape:", X_pca.shape)
np.savetxt('phishing_pca.csv', X_pca)

pca = PCA(0.9)
pca.fit(optdigits_X)
# print(pca.components_)
# print(pca.explained_variance_)

X_pca = pca.transform(optdigits_X)
print("original shape:   ", optdigits_X.shape)
print("transformed shape:", X_pca.shape)
np.savetxt('optical_pca.csv', X_pca)



# Variance plots


# pca = PCA().fit(phishing_X)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('Phishing: Cumulative Explained Variance vs Number of Components')
# plt.show()

# pca = PCA().fit(optdigits_X)
# plt.plot(np.cumsum(pca.explained_variance_ratio_))
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('Optical Digits: Cumulative Explained Variance vs Number of Components')
# plt.show()


# Eigenvalues plots

# pca = PCA()
# pca.fit(phishing_X)
# plt.plot(pca.explained_variance_)
# plt.xlabel('Principal Components')
# plt.ylabel('Eigenvalues')
# plt.title('Phishing: Eigenvalues vs Principal Components')
# plt.show()

# pca = PCA()
# pca.fit(optdigits_X)
# plt.plot(pca.explained_variance_)
# plt.xlabel('Principal Components')
# plt.ylabel('Eigenvalues')
# plt.title('Optical Digits: Eigenvalues vs Principal Components')
# plt.show()





