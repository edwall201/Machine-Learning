import numpy as np
import matplotlib.pyplot as plt

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of sigmoid function
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Initial parameters
x = 1.0
w = 2.0
b = 2.0
y_true = 0.0
eta = 0.15
epochs = 300

# Lists to store the cost at each epoch
costs = []

# Training loop
for epoch in range(epochs):
    # Forward pass
    z = w * x + b
    y_pred = sigmoid(z)
    
    # Compute the loss (MSE)
    cost = 0.5 * (y_pred - y_true) ** 2
    costs.append(cost)
    
    # Backward pass (gradients)
    dL_dy_pred = y_pred - y_true
    dy_pred_dz = sigmoid_derivative(z)
    
    dL_dw = dL_dy_pred * dy_pred_dz * x
    dL_db = dL_dy_pred * dy_pred_dz
    
    # Update weights and bias
    w -= eta * dL_dw
    b -= eta * dL_db
    
    # Print the progress
    if epoch % 50 == 0:
        print(f'Epoch {epoch}, Cost: {cost:.4f}, w: {w:.4f}, b: {b:.4f}, y_pred: {y_pred:.4f}')

# Plotting the cost over epochs
plt.plot(range(epochs), costs)
plt.xlabel('Epoch')
plt.ylabel('Cost')
plt.title('Cost over Epochs')
plt.show()
