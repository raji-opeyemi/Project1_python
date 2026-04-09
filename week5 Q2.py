from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import numpy as np

# load dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv'
df = read_csv(path, header=None)

# split input and output
X, y = df.values[:, :-1], df.values[:, -1]

# ensure float
X = X.astype('float32')

# encode labels
y = LabelEncoder().fit_transform(y)

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# number of features
n_features = X_train.shape[1]

# define model
model = Sequential()
model.add(Dense(20, activation='relu', input_shape=(n_features,)))
model.add(Dense(10, activation='tanh'))
model.add(Dense(8, activation='sigmoid'))
model.add(Dense(3, activation='softmax'))

# compile
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# train
model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)

# save model
model.save("iris_model.keras")

# load model
loaded_model = load_model("iris_model.keras")

# prediction
sample = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)
probs = loaded_model.predict(sample, verbose=0)[0]
predicted_class = argmax(probs)

print("Probabilities:", probs)
print("Predicted class index:", predicted_class)