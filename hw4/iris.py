from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Initialize variables to store accuracy scores
test_accuracies = []

# Repeat the process 10 times
for _ in range(10):
    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Define the parameter grid for grid search
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.01, 0.1, 1, 10]}

    # Perform grid search with SVM classifier
    svm_clf = SVC(kernel='rbf')
    grid_search = GridSearchCV(svm_clf, param_grid, cv=3)
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters from grid search
    best_params = grid_search.best_params_

    # Train SVM classifier with best hyperparameters
    best_svm_clf = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])
    best_svm_clf.fit(X_train, y_train)

    # Evaluate on test set
    y_pred = best_svm_clf.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_accuracies.append(test_accuracy)

# Calculate and print the average accuracy over 10 iterations
average_accuracy = np.mean(test_accuracies)
print(f'Average accuracy over 10 iterations: {average_accuracy:.4f}')
