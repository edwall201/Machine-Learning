import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Filter the dataset to include only virginica (label 2) and versicolor (label 1)
mask = (y == 1) | (y == 2)
X = X[mask]
y = y[mask]

# Convert labels to -1 (versicolor) and 1 (virginica) for perceptron
y = np.where(y == 1, -1, 1)

# Normalize the features for better convergence
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Perceptron training function
def train_perceptron(X_train, y_train, eta, n_iter):
    w = np.zeros(X_train.shape[1] + 1)  # +1 for bias term
    for _ in range(n_iter):
        for xi, target in zip(X_train, y_train):
            update = eta * (target - predict(xi, w))
            w[1:] += update * xi
            w[0] += update  # bias update
    return w

# Perceptron prediction function
def net_input(X, w):
    return np.dot(X, w[1:]) + w[0]

def predict(X, w):
    return np.where(net_input(X, w) >= 0.0, 1, -1)

# Evaluation function
def evaluate_perceptron(eta, n_iter, n_trials=10):
    accuracies = []
    for _ in range(n_trials):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)
        w = train_perceptron(X_train, y_train, eta, n_iter)
        y_pred = np.array([predict(xi, w) for xi in X_test])
        accuracy = np.mean(y_pred == y_test)
        accuracies.append(accuracy)
    return np.mean(accuracies)

# Determine suitable values for eta and n_iter
best_eta = None
best_n_iter = None
best_accuracy = 0

# Try different values for eta and n_iter
for eta in [0.001, 0.01, 0.1, 1.0]:
    for n_iter in [10, 50, 100, 200]:
        accuracy = evaluate_perceptron(eta, n_iter)
        print(f"eta: {eta}, n_iter: {n_iter}, accuracy: {accuracy:.4f}")
        if accuracy > best_accuracy:
            best_eta = eta
            best_n_iter = n_iter
            best_accuracy = accuracy

print(f"Best eta: {best_eta}, Best n_iter: {best_n_iter}, Best accuracy: {best_accuracy:.4f}")
