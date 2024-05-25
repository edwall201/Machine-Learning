import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Training and evaluation
num_trials = 10
epochs = 20
all_accuracies = []

for i in range(num_trials):
    model = build_model()
    model.fit(train_images, train_labels, epochs=epochs, batch_size=64, verbose=0)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
    all_accuracies.append(test_acc)
    print(f'Trial {i + 1} accuracy: {test_acc:.4f}')

average_accuracy = np.mean(all_accuracies)
print(f'Average accuracy over {num_trials} trials: {average_accuracy:.4f}')
