from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 1. Create the data (100 samples as requested)
X, y = make_classification(n_samples=100, random_state=1)

# 2. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# 3. USE THE IMPORT: Create the model
# As soon as you type this line, the "Unused import" warning goes away!
model = MLPClassifier(random_state=1, max_iter=300)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Display the results requested by the PDF
print("Probabilities P(y|x):", model.predict_proba(X_test[:1]))
print("Score:", model.score(X_test, y_test))
print("Loss Function:", model.loss_)
print("Activation Function:", model.activation)

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor # Fixed from 'MLPRegression' in PDF

# 1. Generate data (200 samples) [cite: 11]
X, y = make_regression(n_samples=200, random_state=1)

# 2. Split data [cite: 10]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# 3. Create and Train
regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)

# 4. Display results [cite: 13]
print("Prediction Results:", regr.predict(X_test[:1]))
print("Score:", regr.score(X_test, y_test))
print("Loss Function:", regr.loss_)