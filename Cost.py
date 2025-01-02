import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
def generate_synthetic_data(size, noise_std=0.3):
    x = np.random.uniform(0, 1, size)
    noise = np.random.normal(0, noise_std, size)
    y = np.sin(2 * np.pi * x) + noise
    return x, y

# Manual implementation of train_test_split
def train_test_split_manual(x, y, test_size=0.2):
    n = len(x)
    test_n = int(n * test_size)
    indices = np.random.permutation(n)
    test_indices = indices[:test_n]
    train_indices = indices[test_n:]
    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return x_train, x_test, y_train, y_test

# Gradient Descent implementation for both MSE and MAE
def gradient_descent(x, y, lr, cost_function='MSE', iterations=1000):
    m, b = 0.0, 0.0  # Initialize slope and intercept
    n = len(x)
    for _ in range(iterations):
        y_pred = m * x + b
        if cost_function == 'MSE':
            # MSE Gradient Calculation
            dm = (-2/n) * np.sum(x * (y - y_pred))
            db = (-2/n) * np.sum(y - y_pred)
        elif cost_function == 'MAE':
            # MAE Gradient Calculation
            dm = (-1/n) * np.sum(np.sign(y - y_pred) * x)
            db = (-1/n) * np.sum(np.sign(y - y_pred))
        
        # Update parameters
        m -= lr * dm
        b -= lr * db
    
    return m, b

# RMSE Calculation
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Experiment with different learning rates
learning_rates = [0.025, 0.05, 0.1, 0.2, 0.5]
cost_functions = ['MSE', 'MAE']
dataset_size = 1000  # You can experiment with different sizes

# Generate synthetic data
x, y = generate_synthetic_data(dataset_size)

# Split data into training and test sets manually
x_train, x_test, y_train, y_test = train_test_split_manual(x, y, test_size=0.2)

# For storing RMSE values
rmse_results = {cost_fn: [] for cost_fn in cost_functions}

# Experiment for each cost function and learning rate
for cost_fn in cost_functions:
    for lr in learning_rates:
        m, b = gradient_descent(x_train, y_train, lr, cost_function=cost_fn)
        
        # Predict on test set
        y_test_pred = m * x_test + b
        error = rmse(y_test, y_test_pred)
        
        # Store the RMSE
        rmse_results[cost_fn].append(error)

# Plot the RMSE vs Learning Rate
for cost_fn in cost_functions:
    plt.plot(learning_rates, rmse_results[cost_fn], label=f'{cost_fn} Cost')

plt.xlabel('Learning Rate')
plt.ylabel('Test RMSE')
plt.title('RMSE vs Learning Rate for Different Cost Functions')
plt.legend()
plt.grid(True)
plt.show()
