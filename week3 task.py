import numpy as np
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# 1. Load Data [cite: 23, 24]
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
df = read_csv(path, header=None)

# 2. Prepare Inputs (X) and Labels (y) [cite: 26, 28, 30]
X, y = df.values[:, :-1].astype('float32'), df.values[:, -1]
y = LabelEncoder().fit_transform(y)

# 3. Split Data [cite: 32]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# 4. Build the Sequential Model [cite: 38]
n_features = X_train.shape[1]
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(n_features,))) # Hidden Layer
model.add(Dense(3, activation='softmax')) # Output Layer (3 flower types)

# 5. Compile and Train [cite: 39]
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=150, batch_size=32, verbose=0)

# 6. Predict specific row [cite: 35, 36, 37]
row2 = np.array([[5.1, 3.5, 1.4, 0.2]])
yhat = model.predict(row2)
print(f'Predicted: {yhat} (class={argmax(yhat)})')

from pandas import read_csv
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# 1. Load Data [cite: 50, 51]
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv'
df = read_csv(path, header=None)

# 2. Split Data [cite: 53, 55]
X, y = df.values[:, :-1].astype('float32'), df.values[:, -1].astype('float32')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# 3. Build Model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(1)) # Only 1 node for price

# 4. Compile and Train
model.compile(optimizer='adam', loss='mse') # mse = Squared Sum Error [cite: 63]
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

# 5. Predict specific row [cite: 57, 60, 61]
row2 = np.array([[0.00632, 18.0, 2.31, 0, 0.538, 6.575, 65.2, 4.09, 1, 296.0, 15.3, 396.9, 4.98]])
yhat = model.predict(row2)
print(f'Predicted Price: {yhat[0][0]:.3f}') [cite: 61]