from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import adagrad
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nolearn.lasagne import NeuralNet
from util import *

X1, Y1, X2, Y2 = import_data()
X_train, X_test, y_train, y_test = train_test_split(X2, Y2, test_size=0.3)


# Now we prep the data for a neural net
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train).astype(np.int32)
y_test = encoder.fit_transform(y_test).astype(np.int32)
num_classes = len(encoder.classes_)
num_features = X_train.shape[1]
epochs = 20
val_auc1A, val_auc2A, val_auc3A = np.zeros(epochs), np.zeros(epochs), np.zeros(epochs)
val_auc1B, val_auc2B, val_auc3B = np.zeros(epochs), np.zeros(epochs), np.zeros(epochs)
# Comment out second layer for run time.
layers = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('output', DenseLayer)
           ]
net1 = NeuralNet(layers=layers,
                 input_shape=(None, num_features),
                 dense0_num_units=512,
                 dropout0_p=0.1,
                 # dense1_num_units=256,
                 # dropout1_p=0.1,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 update=adagrad,
                 update_learning_rate=0.04,
                 eval_size=0.0,
                 verbose=0,
                 max_epochs=1)
for i in range(epochs):
    net1.fit(X_train, y_train)
    pred = net1.predict_proba(X_test)[:,1]
    pred = pred.astype(int)
    val_auc1A[i] = 1 - accuracy_score(y_test,pred)
    pred = net1.predict_proba(X_train)[:,1]
    pred = pred.astype(int)
    val_auc1B[i] = 1 - accuracy_score(y_train,pred)

# Add a second layer to the network.
layers = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('output', DenseLayer)
           ]
net2 = NeuralNet(layers=layers,
                 input_shape=(None, num_features),
                 dense0_num_units=512,
                 dropout0_p=0.1,
                 dense1_num_units=256,
                 dropout1_p=0.1,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 update=adagrad,
                 update_learning_rate=0.04,
                 eval_size=0.0,
                 verbose=0,
                 max_epochs=1)
for i in range(epochs):
    net2.fit(X_train, y_train)
    pred = net2.predict_proba(X_test)[:,1]
    pred = pred.astype(int)
    val_auc2A[i] = 1 - accuracy_score(y_test,pred)
    pred = net2.predict_proba(X_train)[:,1]
    pred = pred.astype(int)
    val_auc2B[i] = 1 - accuracy_score(y_train,pred)

# Add a third layer to the network.
layers = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout0', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
          ('dense2', DenseLayer),
          ('dropout2', DropoutLayer),
            ('dense3', DenseLayer),
                      ('dropout3', DropoutLayer),
            ('dense4', DenseLayer),
                      ('dropout4', DropoutLayer),
            ('dense5', DenseLayer),
          ('dropout5', DropoutLayer),
           ('output', DenseLayer)
           ]
net3 = NeuralNet(layers=layers,
                 input_shape=(None, num_features),
                 dense0_num_units=512,
                 dropout0_p=0.1,
                 dense1_num_units=256,
                 dropout1_p=0.1,
                 dense2_num_units=256,
                 dropout2_p=0.1,
dense3_num_units=128,
                 dropout3_p=0.1,
dense4_num_units=64,
                 dropout4_p=0.1,
dense5_num_units=32,
                 dropout5_p=0.1,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 update=adagrad,
                 update_learning_rate=0.04,
                 eval_size=0.0,
                 verbose=0,
                 max_epochs=1)
for i in range(epochs):
    net3.fit(X_train, y_train)
    pred = net3.predict_proba(X_test)[:,1]
    pred = pred.astype(int)
    val_auc3A[i] = 1 - accuracy_score(y_test,pred)
    pred = net2.predict_proba(X_train)[:,1]
    pred = pred.astype(int)
    val_auc3B[i] = 1 - accuracy_score(y_train,pred)

from matplotlib import pyplot
pyplot.plot(val_auc1A, linewidth=3, label="Test Set : Single layer")
pyplot.plot(val_auc2A, linewidth=3, label="Test Set : Second layer introduced")
pyplot.plot(val_auc3A, linewidth=3, label="Test Set : Five layers introduced")
pyplot.plot(val_auc1B, linewidth=3, label="Train Set : Single layer")
pyplot.plot(val_auc2B, linewidth=3, label="Train Set : Second layer introduced")
pyplot.plot(val_auc3B, linewidth=3, label="Train Set : Five layers introduced")
plt.legend(loc='upper right')
pyplot.xlabel("Epochs (Back Propagation Loop)")
pyplot.ylabel("Error Rate")
plt.title('Optical Digits Dataset')
pyplot.show()