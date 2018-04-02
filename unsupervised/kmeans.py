from sklearn.cluster import KMeans
from sklearn import metrics

from util import *

phishing_X, phishing_Y, optdigits_X, optdigits_Y = import_data()

kmeans = KMeans(n_clusters=2)
kmeans.fit(phishing_X)
phishing_kmeans_Y = kmeans.predict(phishing_X)

kmeans_2= KMeans(n_clusters=10)
kmeans_2.fit(optdigits_X)
optdigits_kmeans_Y = kmeans_2.predict(optdigits_X)

print metrics.adjusted_rand_score(phishing_Y, phishing_kmeans_Y)
print metrics.adjusted_rand_score(optdigits_Y, optdigits_kmeans_Y)



