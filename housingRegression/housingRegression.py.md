```python
import numpy as np
import matplotlib.pyplot as plt

# read in the data
print("linear regressor for estimating Boston housing costs")

training = np.genfromtxt('housing_training.csv', delimiter=',')
training_X = training[:, :-1]
training_Y = training[:,-1]
print("training shape: {}, {}".format(training.shape[0], training.shape[1]))
test = np.genfromtxt('housing_test.csv', delimiter=',')
test_X = test[:, :-1]
test_Y = test[:,-1]
print("test shape: {}, {}".format(test.shape[0], test.shape[1]))


# # Hyper Parameters
learning_rate = 1e-8 # play with this to get different results?
step_iterations = 75 # how many gradient descent steps to take

num_features = len(training_X[0])

def linear_regression(X, y):
    return np.dot(np.dot(np.linalg.pinv(np.dot(X.transpose(), X)), X.transpose()), y)

def cost(X, y, b): # taken from Dr. Kang's lecture code
    return np.square(np.sum(np.dot(X, b) - np.array(y)))

def calculate_gradient_descent_step(X, y, b):
    return -np.dot(X.transpose(), y) + np.dot(np.dot(X.transpose(), X), b)

def predict(X, b):
    return np.dot(X, b)

def calculate_rmse(predictions, Y):
    return (sum((Y - predictions)**2)/len(Y))**.5

b_optimal = linear_regression(training_X, training_Y)
linear_predictions = predict(test_X, b_optimal)
linear_regression_rmse = calculate_rmse(linear_predictions, test_Y)
print("linear regression rmse: {}".format(linear_regression_rmse))
print("estimated weights: {}".format(b_optimal))

b_estimate = np.zeros(num_features)
costs = []



print("learning rate for gradient descent: {} with {} steps".format(learning_rate, step_iterations))
for i in range(0, step_iterations):
    b_estimate = b_estimate - learning_rate * calculate_gradient_descent_step(training_X, training_Y, b_estimate)
    error = calculate_rmse(predict(training_X, b_estimate), training_Y)
    costs.append(error)

gradient_predictions = predict(test_X, b_estimate)
print("gradient descent linear model rmse {}".format(calculate_rmse(gradient_predictions, test_Y)))
print("gradient descent weights: {}".format(b_estimate))

# plt.scatter(linear_predictions, test_Y, color="green")
plt.scatter(gradient_predictions, test_Y, color="red")
plt.title("Boston Housing Prices (in thousands)")
plt.ylabel("Ground Truth")
plt.xlabel("Prediction")
plt.plot([min(test_Y), max(test_Y)], [min(test_Y), max(test_Y)])
plt.show()

```

