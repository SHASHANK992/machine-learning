from sklearn import svm
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from util import import_data
import numpy as np

X1, Y1, X2, Y2 = import_data()

def plot_dataset_1():
    X_train, X_test, y_train, y_test = train_test_split( X1, Y1, test_size = 0.3)

    costs = np.power(10.0, range(-2,2))
    kernels = ['linear', 'poly', 'rbf']
    auc_test = np.zeros((len(costs),len(kernels)))
    auc_train = np.zeros((len(costs),len(kernels)))

    # Comment out second layer for run time.
    for i in range(len(costs)):
        for k in range(len(kernels)):
            svc = svm.SVC(kernel = kernels[k], C=costs[i], probability=True)
            svc.fit(X_train,y_train)
            y_pred_test = svc.predict(X_test)
            y_pred_train = svc.predict(X_train)
            auc_test[i,k] = accuracy_score(y_test, y_pred_test)
            auc_train[i,k] = accuracy_score(y_train, y_pred_train)

    for k in range(len(kernels)):
        pyplot.plot(auc_test[:,k], linewidth=1, label="Test Accuracy  : "+kernels[k])
    for k in range(len(kernels)):
        pyplot.plot(auc_train[:,k], linewidth=1, label="Train Accuracy : "+kernels[k])

    plt.legend(loc='lower right')
    pyplot.ylim(0.8, 1.0)
    np.set_printoptions(precision=3)
    plt.xticks(range(len(costs)),['0.001', '0.01','0.1', '1','10', '100','1000'])
    pyplot.xlabel("Costs")
    pyplot.ylabel("Accuracy")
    pyplot.title("Phising Dataset")
    pyplot.show()


def plot_dataset_2():
    X_train, X_test, y_train, y_test = train_test_split(X2, Y2, test_size=0.3)

    costs = np.power(10.0, range(-2, 2))
    kernels = ['poly', 'rbf']
    acc_test = np.zeros((len(costs), len(kernels)))
    acc_train = np.zeros((len(costs), len(kernels)))

    # Comment out second layer for run time.
    for i in range(len(costs)):
        for k in range(len(kernels)):
            svc = svm.SVC(kernel=kernels[k], C=costs[i], probability=True)
            svc.fit(X_train, y_train)
            y_pred_test = svc.predict(X_test)
            y_pred_train = svc.predict(X_train)
            acc_test[i, k] = f1_score(y_test, y_pred_test, average='macro')
            acc_train[i, k] = f1_score(y_train, y_pred_train, average='macro')

    for k in range(len(kernels)):
        pyplot.plot(acc_test[:, k], linewidth=1, label="Test F-1 Score  : " + kernels[k])
    for k in range(len(kernels)):
        pyplot.plot(acc_train[:, k], linewidth=1, label="Train F-1 Score : " + kernels[k])

    plt.legend(loc='lower right')
    # pyplot.ylim(0.75, 1.1)
    np.set_printoptions(precision=3)
    plt.xticks(range(len(costs)), ['0.001', '0.01', '0.1', '1', '10', '100', '1000'])
    pyplot.xlabel("Costs")
    pyplot.ylabel("Accuracy")
    pyplot.title("Optical Digits Dataset")
    pyplot.show()

# plot_dataset_1()
plot_dataset_2()