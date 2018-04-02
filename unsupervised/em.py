from sklearn.mixture import GMM
import numpy

from util import *
from sklearn import metrics

phishing_X, phishing_Y, optdigits_X, optdigits_Y = import_data_pca()

scores = []
# scores.append(['No of Clusters', 'Adjusted Rand Score', 'Adjusted Mutual Info Score', 'Homogeneity Score', 'Completeness Score', \
#                'V Measure Score' ])

for i in range(1, 10):

    gmm = GMM(n_components=i)
    gmm.fit(phishing_X)
    phishing_kmeans_Y = gmm.predict(phishing_X)

    scores.append([i, metrics.adjusted_rand_score(phishing_Y, phishing_kmeans_Y) \
                  ,metrics.adjusted_mutual_info_score(phishing_Y, phishing_kmeans_Y) \
                  ,metrics.homogeneity_score(phishing_Y, phishing_kmeans_Y) \
                  ,metrics.completeness_score(phishing_Y, phishing_kmeans_Y) \
                  ,metrics.v_measure_score(phishing_Y, phishing_kmeans_Y)])

numpy.savetxt("scores_optdigits.csv", scores, delimiter=",")



