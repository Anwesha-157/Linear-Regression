import numpy as np
import matplotlib.pyplot as plt

# Function to generate synthetic dataset
def generate_synthetic_data(size, noise_std=0.3):
    x = np.random.uniform(0, 1, size)
    noise = np.random.normal(0, noise_std, size)
    y = np.sin(2 * np.pi * x) + noise
    return x, y

# Manually split data into train and test sets
def train_test_split_manual(x, y, test_size=0.2):
    n = len(x)
    test_n = int(n * test_size)
    indices = np.random.permutation(n)
    test_indices = indices[:test_n]
    train_indices = indices[test_n:]
    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return x_train, x_test, y_train, y_test

# Function to generate polynomial features
def polynomial_features(x, degree):
    x_poly = np.ones((len(x), 1))  # Add a column of 1s for the intercept
    for d in range(1, degree + 1):
        x_poly = np.hstack((x_poly, x**d))
    return x_poly

# Linear regression using gradient descent
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, iterations=10000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.coefficients = None

    def fit(self, X, y):
        n, m = X.shape
        self.coefficients = np.zeros(m)
        for _ in range(self.iterations):
            y_pred = X.dot(self.coefficients)
            gradient = -(2/n) * X.T.dot(y - y_pred)
            self.coefficients -= self.learning_rate * gradient

    def predict(self, X):
        return X.dot(self.coefficients)

# Mean Squared Error function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Function to fit polynomial regression and calculate train and test errors
def fit_and_calculate_errors(x_train, y_train, x_test, y_test, degree):
    x_train_poly = polynomial_features(x_train.reshape(-1, 1), degree)
    x_test_poly = polynomial_features(x_test.reshape(-1, 1), degree)

    model = LinearRegressionGD(learning_rate=0.01, iterations=10000)
    model.fit(x_train_poly, y_train)

    y_train_pred = model.predict(x_train_poly)
    y_test_pred = model.predict(x_test_poly)

    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)

    return train_error, test_error

# Dataset sizes
dataset_sizes = [10, 100, 1000, 10000]
degrees = range(1, 10)

# Store train and test errors for each dataset size
all_train_errors = []
all_test_errors = []

for size in dataset_sizes:
    x, y = generate_synthetic_data(size)

    # Split into training and test sets
    x_train, x_test, y_train, y_test = train_test_split_manual(x, y)

    train_errors = []
    test_errors = []

    for degree in degrees:
        train_error, test_error = fit_and_calculate_errors(x_train, y_train, x_test, y_test, degree)
        train_errors.append(train_error)
        test_errors.append(test_error)

    all_train_errors.append(train_errors)
    all_test_errors.append(test_errors)

# 3. Plot learning curves for train and test errors
plt.figure(figsize=(10, 6))
for i, size in enumerate(dataset_sizes):
    plt.plot(degrees, all_train_errors[i], label=f'Train Error (size={size})', marker='o')
    plt.plot(degrees, all_test_errors[i], label=f'Test Error (size={size})', linestyle='--', marker='x')

plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curves: Train and Test Errors for Different Dataset Sizes')
plt.legend()
plt.grid(True)
plt.show()

# 4. Learning curve for dataset size vs. errors
train_errors_vs_size = []
test_errors_vs_size = []

for i in range(len(dataset_sizes)):
    # Select the best polynomial degree for each dataset (degree 3 or 4 tends to work well)
    best_train_error = min(all_train_errors[i])
    best_test_error = min(all_test_errors[i])

    train_errors_vs_size.append(best_train_error)
    test_errors_vs_size.append(best_test_error)

# Plot dataset size vs. best train/test error
plt.figure(figsize=(8, 5))
plt.plot(dataset_sizes, train_errors_vs_size, label='Train Error', marker='o', color='blue')
plt.plot(dataset_sizes, test_errors_vs_size, label='Test Error', marker='x', color='red')

plt.xscale('log')
plt.xlabel('Dataset Size (log scale)')
plt.ylabel('Mean Squared Error')
plt.title('Learning Curve: Train and Test Errors vs. Dataset Size')
plt.legend()
plt.grid(True)
plt.show()
