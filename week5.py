from sklearn.datasets import make_classification
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
import numpy as np

# create dataset
X, y = make_classification(
    n_samples=1000,
    n_features=4,
    n_classes=2,
    random_state=1
)

# number of input features
n_features = X.shape[1]

# define model
model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(n_features,)))
model.add(Dense(1, activation='sigmoid'))

# compile model
model.compile(optimizer=SGD(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# train model
model.fit(X, y, epochs=50, batch_size=32, verbose=0)

# save model
model.save("model.keras")

# load model
loaded_model = load_model("model.keras")

# prediction
sample = np.array([[1.91518414, 1.14995454, -1.52847073, 0.79430654]], dtype=np.float32)
prediction = loaded_model.predict(sample, verbose=0)

print("Predicted probability:", prediction[0][0])
print("Predicted class:", int(prediction[0][0] >= 0.5))

