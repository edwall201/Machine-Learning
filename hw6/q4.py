import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
# Load and preprocess the data
iris_data = load_iris()
x = iris_data.data
y_ = iris_data.target.reshape(-1, 1) # Convert data to a single column

# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)

# Split the data for training and testing
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)

# Initialize a list to store average accuracies
avg_accuracies = []

# For each number of hidden units from 10 to 100
for units in range(10, 101, 10):
    # Initialize a list to store accuracies for each experiment
    accuracies = []
    # Repeat the experiment 10 times
    for _ in range(10):
        # Define the model
        model = Sequential()
        model.add(Dense(units, input_shape=(4,), activation='relu'))
        model.add(Dense(units, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        # Compile the model
        model.compile(Adam(lr=0.04), 'categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(train_x, train_y, verbose=0, batch_size=5, epochs=10)

        # Evaluate the model
        _, accuracy = model.evaluate(test_x, test_y, verbose=0)
        accuracies.append(accuracy)

    # Calculate and store the average accuracy
    avg_accuracy = np.mean(accuracies)
    avg_accuracies.append(avg_accuracy)
    print(f'Average accuracy for {units} hidden units: {avg_accuracy}')

print('Average accuracies for each number of hidden units:', avg_accuracies)