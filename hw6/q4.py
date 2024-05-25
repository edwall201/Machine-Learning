import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam

iris_data = load_iris()
x = iris_data.data
y_ = iris_data.target.reshape(-1, 1) 
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_)
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.20)
avg_accuracies = []

for units in range(10, 101, 10):
    accuracies = []
    for _ in range(10):
        model = Sequential()
        model.add(Dense(units, input_shape=(4,), activation='relu'))
        model.add(Dense(units, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        model.compile(Adam(learning_rate=0.04), 'categorical_crossentropy', metrics=['accuracy'])   
        model.fit(train_x, train_y, verbose=0, batch_size=5, epochs=10)
        _, accuracy = model.evaluate(test_x, test_y, verbose=0)
        accuracies.append(accuracy)
    avg_accuracy = np.mean(accuracies)
    avg_accuracies.append(avg_accuracy)
    print(f'Average accuracy for {units} hidden units: {avg_accuracy}')

print('Average accuracies for each number of hidden units:', avg_accuracies)