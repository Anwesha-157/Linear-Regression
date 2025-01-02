import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset
np.random.seed(42)
x = np.random.uniform(0, 1, 10)
noise = np.random.normal(0, 0.3, x.shape)
y = np.sin(2 * np.pi * x) + noise

# Split into training and test sets (80% train, 20% test)
def train_test_split_manual(x, y, test_size=0.2):
    n = len(x)
    test_n = int(n * test_size)
    indices = np.random.permutation(n)
    test_indices = indices[:test_n]
    train_indices = indices[test_n:]
    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return x_train, x_test, y_train, y_test

x_train, x_test, y_train, y_test = train_test_split_manual(x, y)

# Reshape for consistency with polynomial fitting
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)

# Function to generate polynomial features
def polynomial_features(x, degree):
    x_poly = np.ones((len(x), 1))  # Add column of 1s for the intercept
    for d in range(1, degree + 1):
        x_poly = np.hstack((x_poly, x**d))
    return x_poly

# Linear Regression using Gradient Descent
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, iterations=1000):
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

# Function to calculate Mean Squared Error (MSE)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Visualizing Dataset
plt.scatter(x, y, color='blue', label='Synthetic Data')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Synthetic Data with Noise')
plt.legend()
plt.show()

# Fit polynomial models for degrees 1 to 9
degrees = range(1, 10)
train_errors = []
test_errors = []

plt.scatter(x, y, color='blue', label='Synthetic Data')
for degree in degrees:
    # Generate polynomial features
    x_train_poly = polynomial_features(x_train, degree)
    x_test_poly = polynomial_features(x_test, degree)
    
    # Initialize and fit the model
    model = LinearRegressionGD(learning_rate=0.01, iterations=10000)
    model.fit(x_train_poly, y_train)
    
    # Make predictions
    y_train_pred = model.predict(x_train_poly)
    y_test_pred = model.predict(x_test_poly)
    
    # Calculate errors
    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)
    
    train_errors.append(train_error)
    test_errors.append(test_error)
    
    # Plot fitted curve for this degree
    x_curve = np.linspace(0, 1, 100).reshape(-1, 1)
    x_curve_poly = polynomial_features(x_curve, degree)
    y_curve = model.predict(x_curve_poly)
    plt.plot(x_curve, y_curve, label=f'Degree {degree}')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitted Curves for Different Polynomial Degrees')
plt.legend()
plt.show()

# 2(b) Plot train and test errors for different values of n (degree of polynomial)
plt.plot(degrees, train_errors, label='Train Error', color='blue', marker='o')
plt.plot(degrees, test_errors, label='Test Error', color='red', marker='o')
plt.xlabel('Polynomial Degree (n)')
plt.ylabel('Mean Squared Error')
plt.title('Train and Test Errors vs. Polynomial Degree')
plt.legend()
plt.show()

# Determine suitable value of n based on test error
suitable_n = np.argmin(test_errors) + 1  # Index of minimum test error + 1 (since degree starts at 1)
print(f"The most suitable degree for this dataset is: {suitable_n}")
