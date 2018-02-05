import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.neural_network import MLPClassifier

from util import import_data

X1, Y1, X2, Y2 = import_data()

def gridSearchCV(X_train, y_train, hidden_layer_sizes):
    gs = GridSearchCV(estimator=MLPClassifier(activation='relu',
           hidden_layer_sizes=hidden_layer_sizes, learning_rate='constant',
           max_iter=300, random_state=100, solver='sgd'), param_grid={
        'learning_rate_init': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        'momentum': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}, cv=5)

    gs.fit(X_train, y_train)
    print gs.best_params_
    print gs.grid_scores_
    return gs.best_params_['momentum'], gs.best_params_['learning_rate_init']

def getEpochCurves(momentum1, learning_rate1, momentum2, learning_rate2, X_train, X_test, y_train, y_test, title, auc):
    #Testing for various epoch results with 1 hidden layer
    fig = plt.figure()
    ax = fig.add_subplot(111)
    accuracy_test_1 = []
    accuracy_train_1 = []
    auc_test_1 = []
    auc_train_1 = []
    for i in range(5, 300, 5):
        classifier = MLPClassifier(activation='relu', hidden_layer_sizes=(nodes,), learning_rate='constant',
                      max_iter=i, random_state=100, warm_start=False, momentum=momentum1, \
                                   learning_rate_init=learning_rate1, solver='sgd')
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_pred_train = classifier.predict(X_train)
        accuracy_test_1.append(accuracy_score(y_test, y_pred))
        accuracy_train_1.append(accuracy_score(y_train, y_pred_train))
        if auc:
            auc_test_1.append(roc_auc_score(y_test, y_pred))
            auc_train_1.append(roc_auc_score(y_train, y_pred_train))

    #Testing for various epoch results with 2 hidden layers
    accuracy_test_2 = []
    accuracy_train_2 = []
    auc_test_2 = []
    auc_train_2 = []
    for i in range(5, 300, 5):
        classifier = MLPClassifier(activation='relu', hidden_layer_sizes=(nodes,nodes), learning_rate='constant',
                      max_iter=i, random_state=100, warm_start=False, momentum=momentum2, \
                                   learning_rate_init=learning_rate2, solver='sgd')
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        y_pred_train = classifier.predict(X_train)
        accuracy_test_2.append(accuracy_score(y_test, y_pred))
        accuracy_train_2.append(accuracy_score(y_train, y_pred_train))
        if auc:
            auc_test_2.append(roc_auc_score(y_test, y_pred))
            auc_train_2.append(roc_auc_score(y_train, y_pred_train))

    ax.plot(range(5, 300, 5), accuracy_test_1, label='Test Accuracy with 1 Hidden Layer')
    ax.plot(range(5, 300, 5), accuracy_train_1, label='Training Accuracy with 1 Hidden Layer')
    ax.plot(range(5, 300, 5), accuracy_test_2, label='Test Accuracy with 2 Hidden Layers')
    ax.plot(range(5, 300, 5), accuracy_train_2, label='Training Accuracy with 2 Hidden Layers')
    if auc:
        ax.plot(range(5, 300, 5), auc_test_1, label='Test AUC with 1 Hidden Layer')
        ax.plot(range(5, 300, 5), auc_train_1, label='Training AUC with 1 Hidden Layer')
        ax.plot(range(5, 300, 5), auc_test_2, label='Test AUC with 2 Hidden Layers')
        ax.plot(range(5, 300, 5), auc_train_2, label='Training AUC with 2 Hidden Layers')
    if auc:
        ax.set_ylabel('Accuracy/AUC')
    else:
        ax.set_ylabel('Accuracy')
    ax.set_xlabel('Epoch')
    if auc:
        plt.title('Accuracy & AUC vs epochs for ' + title)
    else:
        plt.title('Accuracy vs epochs for ' + title)
    plt.legend()
    plt.show()

# Phishing Dataset - using 1 hidden layer
X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3)
num_features = X_train.shape[1]
num_classes = 2
nodes = (num_classes + num_features) / 2
momentum1, learning_rate1 = gridSearchCV(X_train, y_train, (nodes,))
# momentum1, learning_rate1 = 0.9, 0.25

# Phishing Dataset - using 2 hidden layer
momentum2, learning_rate2 = gridSearchCV(X_train, y_train, (nodes, nodes))
# momentum2, learning_rate2 = 0.9, 0.3

getEpochCurves(momentum1, learning_rate1, momentum2, learning_rate2, X_train, X_test, y_train, y_test, \
               'Phishing Dataset', True)



# # Optical Digits Dataset - using 1 hidden layer
# X_train, X_test, y_train, y_test = train_test_split( X2, Y2, test_size = 0.3)
# num_features = X_train.shape[1]
# num_classes = 10
# nodes = (num_classes + num_features)/2
# momentum1, learning_rate1 = gridSearchCV(X_train, y_train,  (nodes,))
# print momentum1, learning_rate1
# # momentum1, learning_rate1 = 0.5, 0.3
#
# # Optical Digits Dataset - using 2 hidden layers
# momentum2, learning_rate2 = gridSearchCV(X_train, y_train, (nodes, nodes))
# print momentum2, learning_rate2
# # momentum2, learning_rate2 = 0.8, 0.35
#
# # AUC is false as this is multi class
# getEpochCurves(momentum1, learning_rate1, momentum2, learning_rate2, X_train, X_test, y_train, y_test, \
#                'Optical Digits Dataset', False)