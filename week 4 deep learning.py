def mse(y_true, y_pred):
    total = 0
    for i in range(len(y_true)):
        diff = y_pred[i] - y_true[i]
        total += diff ** 2
    return total / len(y_true)


y_true = [0.000, 0.166, 0.333]
y_pred = [0.000, 0.254, 0.998]
error = mse(y_true, y_pred)
print("MSE Error:", error)


def mse(y_true, y_pred):
    total = 0
    for i in range(len(y_true)):
        diff = y_true[i] - y_pred[i]
        total += abs(diff)
        return total / len(y_true)


y_true = [0.000, 0.166, 0.333]
y_pred = [0.000, 0.254, 0.998]
error = mse(y_true, y_pred)
print("MAE Error:", error)

import math
def cross_entropy(y_true, y_pred):
    samples = len(y_true)
    total = 0
    for i in range(samples):
        for j in range(len(y_true[i])):
            total += (y_true[i][j] * math.log(y_pred[i][j]))
    return -total / samples


y_true = [[0, 0, 0, 1], [0, 0, 0, 1]]

y_pred = [[0.25, 0.25, 0.25, 0.25], [0.01, 0.01, 0.01, 0.96]]
error = cross_entropy(y_true, y_pred)
print("Cross Entropy loss:", error)


from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Model          # <-- Functional model uses Model
from tensorflow.keras.layers import Dense, Input  # <-- Input layer needed
from tensorflow.keras.utils import to_categorical
import numpy as np

# ---- LOAD DATA ----
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
df = read_csv(path, header=None)

X, y = df.values[:, :-1], df.values[:, -1]
X = X.astype('float32')

# Convert text labels ("Iris-setosa" etc.) to numbers (0, 1, 2)
y = LabelEncoder().fit_transform(y)

# Convert numbers to "one-hot" format: 0 -> [1,0,0], 1 -> [0,1,0], 2 -> [0,0,1]
y = to_categorical(y)

# Split 67% train, 33% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# ---- BUILD FUNCTIONAL MODEL ----
n_features = X_train.shape[1]   # = 4 features

# Step 1: Define the input shape
inputs = Input(shape=(n_features,))

# Step 2: Add hidden layers (you connect them like a chain)
x = Dense(10, activation='relu')(inputs)    # 10 neurons, relu activation
x = Dense(8, activation='relu')(x)          # 8 neurons

# Step 3: Output layer — 3 neurons for 3 classes, softmax gives probabilities
outputs = Dense(3, activation='softmax')(x)

# Step 4: Create the model by specifying inputs and outputs
model = Model(inputs=inputs, outputs=outputs)

# ---- COMPILE (configure how it learns) ----
model.compile(
    optimizer='adam',                        # learning algorithm
    loss='categorical_crossentropy',         # loss for multi-class
    metrics=['accuracy']                     # we want to see accuracy
)

# ---- TRAIN ----
model.fit(X_train, y_train, epochs=150, batch_size=5, verbose=1)

# ---- EVALUATE (a) Training accuracy ----
_, accuracy = model.evaluate(X_train, y_train, verbose=0)
print('Training Accuracy: %.2f%%' % (accuracy * 100))

# ---- PREDICT (b) Prediction results ----
row2 = np.array([[5.1, 3.5, 1.4, 0.2]])
yhat = model.predict(row2)
print('Predicted probabilities:', yhat)
print('Predicted class: %d' % argmax(yhat))


from numpy import sqrt
from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input
import numpy as np

# ---- LOAD DATA ----
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(path, header=None)

X, y = df.values[:, :-1], df.values[:, -1]
X = X.astype('float32')
y = y.astype('float32')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# ---- BUILD FUNCTIONAL MODEL ----
n_features = X_train.shape[1]   # = 13 features

inputs = Input(shape=(n_features,))
x = Dense(64, activation='relu')(inputs)
x = Dense(32, activation='relu')(x)

# Output: single neuron, no activation (we want a raw number, not a probability)
outputs = Dense(1)(x)

model = Model(inputs=inputs, outputs=outputs)

# ---- COMPILE ----
model.compile(
    optimizer='adam',
    loss='mse'       # Mean Squared Error for regression
)

# ---- TRAIN ----
model.fit(X_train, y_train, epochs=150, batch_size=8, verbose=1)

# ---- EVALUATE (c) Squared Sum Error ----
# model.evaluate returns the MSE loss
mse_loss = model.evaluate(X_test, y_test, verbose=0)

# Squared Sum Error = MSE * number of test samples
sse = mse_loss * len(y_test)
print('Squared Sum Error (SSE): %.3f' % sse)

# Also show RMSE (Root Mean Squared Error) — more intuitive
rmse = sqrt(mse_loss)
print('RMSE: %.3f' % rmse)

# ---- PREDICT (d) Prediction results ----
row2 = np.array([[0.00632,18.00,2.310,0,0.5380,6.5750,65.20,
                  4.0900,1,296.0,15.30,396.90,4.98]])
yhat = model.predict(row2)
print('Predicted house price: %.3f' % yhat)