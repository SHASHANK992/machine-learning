import time
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import FastICA
from util import import_data
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111)

def getEpochCurves(momentum1, learning_rate1, X_train, X_test, y_train, y_test, var):

    accuracy_test_1 = []
    accuracy_train_1 = []
    auc_test_1 = []
    auc_train_1 = []
    start, end = 0, 0
    for i in range(5, 200, 5):
        classifier = MLPClassifier(activation='logistic', hidden_layer_sizes=(nodes,), learning_rate='constant',
                      max_iter=i, random_state=100, warm_start=False, momentum=momentum1, \
                                   learning_rate_init=learning_rate1, solver='sgd')
        start = time.time()
        classifier.fit(X_train, y_train)
        end = time.time() - start
        y_pred = classifier.predict(X_test)
        y_pred_train = classifier.predict(X_train)
        accuracy_test_1.append(accuracy_score(y_test, y_pred))
        accuracy_train_1.append(accuracy_score(y_train, y_pred_train))

    ax.plot(range(5, 200, 5), accuracy_test_1, label='Test Accuracy with '+var+' Components')
    ax.plot(range(5, 200, 5), accuracy_train_1, label='Training Accuracy with '+var+' Components')

    return end


# phishing_X, Y1, optdigits_X, Y2 = import_data()
# #
# #
# # Phishing Dataset - using 1 hidden layer
#
# # without PCA
# X1 = phishing_X
#
# X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3)
# num_features = X_train.shape[1]
# num_classes = 2
# nodes = (num_classes + num_features) / 2
# momentum1, learning_rate1 = 0.9, 0.25
#
#
# end = getEpochCurves(momentum1, learning_rate1, X_train, X_test, y_train, y_test, str(X1.shape[1]))
# print "Time taken with " + str(X1.shape[1]) + " components " + str(end)


# with 25 components
phishing_X, Y1, optdigits_X, Y2 = import_data()

pca = FastICA(n_components=26, random_state=5)
X1 = pca.fit_transform(phishing_X)
X1 /= X1.std(axis=0)
print("original shape:   ", phishing_X.shape)
print("transformed shape:", X1.shape)
projected_phishing = np.hstack((X1, Y1[...,None]))
np.savetxt('phishing_ica.csv', projected_phishing, delimiter=',')

# X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3)
# num_features = X_train.shape[1]
# num_classes = 2
# nodes = (num_classes + num_features) / 2
# momentum1, learning_rate1 = 0.9, 0.25
#
#
# end = getEpochCurves(momentum1, learning_rate1, X_train, X_test, y_train, y_test, str(X1.shape[1]))
# print "Time taken with " + str(X1.shape[1]) + " components " + str(end)

#
# # with 28 components
# phishing_X, Y1, optdigits_X, Y2 = import_data()
#
# pca = FastICA(n_components=26)
# pca.fit(phishing_X)
# X1 = pca.transform(phishing_X)
# print("original shape:   ", phishing_X.shape)
# print("transformed shape:", X1.shape)
# projected_phishing = np.hstack((X1, Y1[...,None]))
# np.savetxt('phishing_pca.csv', projected_phishing, delimiter=',')
#
#
# X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3)
# num_features = X_train.shape[1]
# num_classes = 2
# nodes = (num_classes + num_features) / 2
# momentum1, learning_rate1 = 0.9, 0.25
#
#
# end = getEpochCurves(momentum1, learning_rate1, X_train, X_test, y_train, y_test, str(X1.shape[1]))
# print "Time taken with " + str(X1.shape[1]) + " components " + str(end)

# ax.set_ylabel('Accuracy')
# ax.set_xlabel('Epoch')
# plt.title('Accuracy vs epochs for ' + 'Phishing Dataset')
# plt.legend()
# plt.show()



# # Optical Digits Dataset - using 1 hidden layer

# phishing_X, Y1, optdigits_X, Y2 = import_data()
#
# # Without PCA
# X2 = optdigits_X
# X_train, X_test, y_train, y_test = train_test_split( X2, Y2, test_size = 0.3)
# num_features = X_train.shape[1]
# num_classes = 10
# nodes = (num_classes + num_features)/2
# momentum1, learning_rate1 = 0.5, 0.3
#
# end = getEpochCurves(momentum1, learning_rate1, X_train, X_test, y_train, y_test, str(X2.shape[1]))
# print "Time taken with " + str(X2.shape[1]) + " components " + str(end)

# # 95% variance
# phishing_X, Y1, optdigits_X, Y2 = import_data()
#
# pca = FastICA(n_components=46)
# pca.fit(optdigits_X)
#
# X2 = pca.transform(optdigits_X)
# print("original shape:   ", optdigits_X.shape)
# print("transformed shape:", X2.shape)
#
# X_train, X_test, y_train, y_test = train_test_split( X2, Y2, test_size = 0.3)
# num_features = X_train.shape[1]
# num_classes = 10
# nodes = (num_classes + num_features)/2
# momentum1, learning_rate1 = 0.5, 0.3
#
# end = getEpochCurves(momentum1, learning_rate1, X_train, X_test, y_train, y_test, str(X2.shape[1]))
# print "Time taken with " + str(X2.shape[1]) + " components " + str(end)
#
# # 99% variance
phishing_X, Y1, optdigits_X, Y2 = import_data()

pca = FastICA(n_components=51, random_state=5)


X2 = pca.fit_transform(optdigits_X)
X2 /= X2.std(axis=0)
print("original shape:   ", optdigits_X.shape)
print("transformed shape:", X2.shape)
projected_optdigits = np.hstack((X2, Y2[...,None]))
np.savetxt('optical_ica.csv', projected_optdigits, delimiter=',')
#
#
#
# X_train, X_test, y_train, y_test = train_test_split( X2, Y2, test_size = 0.3)
# num_features = X_train.shape[1]
# num_classes = 10
# nodes = (num_classes + num_features)/2
# momentum1, learning_rate1 = 0.5, 0.3
#
# end = getEpochCurves(momentum1, learning_rate1, X_train, X_test, y_train, y_test, str(X2.shape[1]))
#
# print "Time taken with " + str(X2.shape[1]) + " components " + str(end)
#
# ax.set_ylabel('Accuracy')
# ax.set_xlabel('Epoch')
# plt.title('Accuracy vs epochs for ' + 'Optical Digits Dataset')
# plt.legend()
# plt.show()