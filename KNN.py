from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split, ShuffleSplit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from util import *
from sklearn.neighbors import KNeighborsClassifier

X1, Y1, X2, Y2 = import_data()

def testKParam(X, Y, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    accuracy_test = []
    accuracy_train = []
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
    for i in range(1, 21):
        clf_gini = KNeighborsClassifier(n_neighbors=i)
        clf_gini.fit(X_train, y_train)
        y_pred = clf_gini.predict(X_test)
        y_pred_train = clf_gini.predict(X_train)
        accuracy_test.append(accuracy_score(y_test, y_pred) * 100)
        accuracy_train.append(accuracy_score(y_train, y_pred_train) * 100)

    ax.plot(range(1, 21), accuracy_test)
    ax.plot(range(1, 21), accuracy_train)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('K')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def getParametersFromGridSearchCV(X_train, y_train):

    param_grid = {'n_neighbors': np.arange(1, 20)}

    tree = GridSearchCV(estimator = KNeighborsClassifier(), param_grid=param_grid, cv= 10)
    tree.fit(X_train, y_train)
    print tree.best_params_
    return tree.best_params_['n_neighbors']

    # param_grid = {'n_neighbors': np.arange(1, 21)}
    #
    # tree = GridSearchCV(estimator = KNeighborsClassifier(), param_grid=param_grid, cv= 10)
    # tree.fit(X_train, y_train)
    # print tree.best_params_
    # print tree.grid_scores_
    # Cs = [1]
    # Gammas = np.arange(1, 21)
    # scores = [x[1] for x in tree.grid_scores_]
    # scores = np.array(scores).reshape(len(Cs), len(Gammas))
    #
    # for ind, i in enumerate(Cs):
    #     plt.plot(Gammas, scores[ind], label=': ' + str(i))
    # plt.legend()
    # plt.xlabel('K')
    # plt.ylabel('Accuracy')
    # plt.show()
    # return tree.best_params_['n_neighbors']

def draw_learning_curve_1():
    title = "Learning Curve (Phishing)(KNN)"
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.3)

    n_neighbors = getParametersFromGridSearchCV(X_train, y_train)
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    plot_learning_curve(estimator, title, X1, Y1, ylim=None, cv=cv)

    plt.show()

def draw_learning_curve_2():
    title = "Learning Curve (Optical Digits)(KNN)"
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(X2, Y2, test_size=0.3)

    n_neighbors = getParametersFromGridSearchCV(X_train, y_train)
    estimator = KNeighborsClassifier(n_neighbors=n_neighbors)
    plot_learning_curve(estimator, title, X2, Y2, ylim=None, cv=cv)

    plt.show()

testKParam(X2, Y2, 'Optical Digits')