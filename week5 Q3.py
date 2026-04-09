from tensorflow.keras import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import numpy as np

train = [1,2,3,4,5,6,7,8,9,10]

windowSize = 3
X_train, y_train = [], []

for index in range(len(train) - windowSize):
    X_train.append(train[index:index + windowSize])
    y_train.append(train[index + windowSize])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape((len(X_train), windowSize, 1))

def build_model():
    model = Sequential()
    model.add(SimpleRNN(10, input_shape=(windowSize, 1)))
    model.add(Dense(1))  # Linear activation
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

for epochs in [10, 50, 500]:
    print("\nEpochs:", epochs)
    model = build_model()
    model.fit(X_train, y_train, epochs=epochs, verbose=0)

    loss, mae = model.evaluate(X_train, y_train, verbose=0)
    print("MSE:", loss)
    print("MAE:", mae)

    sample = np.array([[2,3,4]], dtype=np.float32).reshape((1, windowSize, 1))
    pred = model.predict(sample, verbose=0)[0][0]
    print("Prediction for [2,3,4]:", pred)