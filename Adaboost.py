from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from util import *

X1, Y1, X2, Y2 = import_data()
X_train, X_test, y_train, y_test = train_test_split( X1, Y1, test_size = 0.3)

#Phishing
# {'max_depth': 19, 'min_samples_leaf': 1}
fig = plt.figure()
ax = fig.add_subplot(111)
accuracy_test_1 = []
accuracy_train_1 = []
for i in range(1, 201):
    classifier_1 = AdaBoostClassifier( DecisionTreeClassifier(max_depth=5, random_state=100, min_samples_leaf=1),
        n_estimators=i, learning_rate=0.5)
    classifier_1.fit(X_train, y_train)
    y_pred = classifier_1.predict(X_test)
    y_pred_train = classifier_1.predict(X_train)
    accuracy_test_1.append(accuracy_score(y_test, y_pred))
    accuracy_train_1.append(accuracy_score(y_train, y_pred_train))

ax.plot(range(1, 201), accuracy_test_1 )
ax.plot(range(1, 201), accuracy_train_1)
ax.set_ylabel('Accuracy')
ax.set_xlabel('Estimators')
plt.title('Estimators vs Accuracy for Phishing Dataset')
plt.tight_layout()
plt.show()





#Phishing
# {'max_depth': 19, 'min_samples_leaf': 1}
# #Optical Digits
# # {'max_depth': 17, 'min_samples_leaf': 1}
# X_train, X_test, y_train, y_test = train_test_split( X2, Y2, test_size = 0.3)
# fig = plt.figure()
# ax = fig.add_subplot(111)
# accuracy_test_2 = []
# accuracy_train_2 = []
# for i in range(1, 201):
#     classifier_2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5, random_state=100, min_samples_leaf=1),
#                                       n_estimators=i, learning_rate=0.5)
#     classifier_2.fit(X_train, y_train)
#     y_pred = classifier_2.predict(X_test)
#     y_pred_train = classifier_2.predict(X_train)
#     accuracy_test_2.append(accuracy_score(y_test, y_pred))
#     accuracy_train_2.append(accuracy_score(y_train, y_pred_train))
#
# ax.plot(range(1, 201), accuracy_test_2 )
# ax.plot(range(1, 201), accuracy_train_2)
# ax.set_ylabel('Accuracy')
# ax.set_xlabel('Estimators')
# plt.title('Estimators vs Accuracy for Optical Digits Dataset')
# plt.tight_layout()
# plt.show()






