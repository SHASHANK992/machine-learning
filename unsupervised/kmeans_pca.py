from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.cluster import k_means_
import matplotlib.pyplot as plt
import numpy

from util import *
from sklearn import metrics

phishing_X, phishing_Y, optdigits_X, optdigits_Y = import_data_pca()

scores = []
# scores.append(['No of Clusters', 'Adjusted Rand Score', 'Adjusted Mutual Info Score', 'Homogeneity Score', 'Completeness Score', \
#                'V Measure Score' ])

for i in range(5,55,5):

    kmeans = k_means_.KMeans(n_clusters=i)
    kmeans.fit(optdigits_X)
    phishing_kmeans_Y = kmeans.predict(optdigits_X)

    scores.append([i, metrics.adjusted_rand_score(optdigits_Y, phishing_kmeans_Y) \
                  ,metrics.adjusted_mutual_info_score(optdigits_Y, phishing_kmeans_Y) \
                  ,metrics.homogeneity_score(optdigits_Y, phishing_kmeans_Y) \
                  ,metrics.completeness_score(optdigits_Y, phishing_kmeans_Y) \
                  ,metrics.v_measure_score(optdigits_Y, phishing_kmeans_Y)])

numpy.savetxt("scores_optdigits.csv", scores, delimiter=",")