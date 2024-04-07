import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

iris = load_iris()
X, y = iris.data[:,:4], iris.target

# Split data into training (70%) and test (30%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
accuracies = {1:[], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[]}
score = 0
for i in range(1,11):
    # Initialize PCA separately for each trial
    pca = PCA(n_components=3)
    X_train_pca = pca.fit_transform(X_train)

    # Train 5-NN
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train_pca, y_train)

    # Transform test set using the PCA fit on the training data
    X_test_pca = pca.transform(X_test)

    # Predict and calculate accuracy
    y_pred = clf.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred)
    accuracies[i].append(acc*100)
    score += acc * 100
# Calculate and print average accuracy across trials
print("Average accuracy after 10 trials:", score/10, "%")

plt.plot(list(accuracies.keys()), list(accuracies.values()))
plt.xlabel("Trial")
plt.ylabel("Accuracy (%)")
# plt.grid(True)
plt.show()
