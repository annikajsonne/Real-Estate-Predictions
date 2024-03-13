import platform, os
print('Python',platform.python_version())
print(platform.system(),os.getcwd())

# Use full canvas width:
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.linear_model import LinearRegression

# Let's change the default font size in the resource configuration. Note the dictionary interface.
matplotlib.rcParams.update({'font.size': 18}) 

# Show all columns when we call a Pandas dataframe
pd.set_option('display.max_columns', None)
import numpy as np
print(f'Using NumPy {np.__version__}')

sales = pd.read_csv('./data/home_data')
N = len(sales)
print(f'There are N={N} records')

def show_features(dataframe, features):
    """Plots columns given by features in the dataframe as histograms."""
    column_num = len(features)
    fig, ax = plt.subplots(column_num, 1, figsize=(14.0, 8.0*column_num), frameon=False)
    if column_num == 1: ax = [ax]
    for i in range(column_num):
        # go through each column of the dataframe and plot the data as a histogram
        column_label = features[i]
        # bin number is minimum of 60 and the number of unique values in the column
        bin_num = min(dataframe[column_label].nunique(), 60)
        dataframe[column_label].hist(ax=ax[i], bins=bin_num)
        # include a table at the right showing statistics for the data
        pd.plotting.table(ax[i], dataframe[column_label].describe(), loc='right', bbox = [1.1, 0.5, 0.25, 0.5])

        ax[i].set_xlabel(column_label)
        ax[i].set_ylabel('count')
        ax[i].margins(0, 0.1)

    plt.show()
    
    return

# show histograms for some of the following features
feature_list = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
       'condition', 'grade', 'sqft_above',
       'sqft_basement', 'yr_built', 'zipcode', 'lat', 'long',
       'sqft_living15', 'sqft_lot15']

# show the features in plot
# show_features(sales, feature_list)

#plot price as a function of sqft_living
fig, ax = plt.subplots(1, 1, figsize=(14.0, 8.0), frameon=False)
ax.plot(sales[1:5000]['sqft_living'], sales[1:5000]['price'], '.')
ax.set_xlabel('sqft_living')
ax.set_ylabel('price')
# plt.show()

# first, extract the price column of the sales DataFrame into a Series
prices = sales['price']

# create the training and test sets
train_set = sales.sample(frac=0.8, random_state=0)
test_set = sales.drop(train_set.index) 

# linear regression closed form solution using matricies
def simple_linear_regression1(in_feature, target):
    '''in_feature: a Pandas series storing the values of the regressor
       target: a Pandas series storing the true values
       Returns: the learned model w=(intercept,slope) using the matrix-based closed-form solution'''
    ...
    X = in_feature.values
    Y = target.values
    
    X_with_intercept = np.vstack((np.ones_like(X), X)).T
    
    weights = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept)).dot(X_with_intercept.T).dot(Y)
    
    intercept, slope = weights[0], weights[1]
    
    return (intercept, slope)

# compute weights using the training set
w1 = simple_linear_regression1(train_set['sqft_living'], train_set['price'])
print(f'Model: {w1[0]:.2f} + {w1[1]:.2f}x')

# linear regression closed form using scikit-learn
def simple_linear_regression2(in_feature, target):
    '''in_feature: a Pandas series storing the values of the regressor
       target: a Pandas series storing the true response values
       Returns: the learned model as a tuple (intercept,slope)'''
    ...
    X = in_feature.values.reshape(-1, 1)
    Y = target.values

    model = LinearRegression()
    model.fit(X, Y)

    intercept = model.intercept_
    slope = model.coef_[0]

    return (intercept, slope)

# compute weights using the training set
w2 = simple_linear_regression2(train_set['sqft_living'], train_set['price'])
print(f'Sklearn model: {w2[0]:.2f} + {w2[1]:.2f}x')

#computes a tuple (X, y) consisting of: An N*2 NumPy matrix X whose columns include a 
#constant column of 1s followed by column for the desired feature, for all N rows of `series`
# and NumPy array $y$ containing the values of `target`.
def series2matrix(series, target):
    X = np.column_stack((np.ones(len(series)), series))
    y = target.values
    
    return X, y

#convert series to matrix
X, y=series2matrix(train_set['sqft_living'],train_set['price'])

# dot product prediction function
def predict(X, w):
    '''X: a matrix whose rows store the tall vectors of the regressor
       w: the model, a tuple with w[0]=intercept and w[1]=slope
       Returns: a vector with the predicted values'''
    return np.dot(X,w)

# mean squared error
def MSE(y,yhat):
    '''y: NumPy array storing the response values
       yhat: Numpy array storing the predicted values
       Returns: the MSE'''
    mse = ((y - yhat) ** 2).mean()
    return mse

#cost of closed form solution
yhat = predict(X, w1)
optimal_cost = MSE(y,yhat)
print(f'Cost of Closed Form Solution: {MSE(y,yhat):.5e}')

# gradient descent function
def simple_linear_regression3(X, y, initial_weights, alpha, tolerance, max_iterations):
    X_min = np.min(X[:, 1:], axis=0)
    X_max = np.max(X[:, 1:], axis=0)
    X_normalized = np.copy(X)
    X_normalized[:, 1:] = (X[:, 1:] - X_min) / (X_max - X_min)
    
    weights = np.array(initial_weights)
    iteration = 0
    gradient_magnitude = tolerance
    cost_history = []

    while iteration < max_iterations:
        predictions = predict(X_normalized, weights)
        error = predictions - y
        gradient = 2 / len(X_normalized) * np.dot(X_normalized.T, error)
        weights = weights - alpha * gradient
        gradient_magnitude = np.linalg.norm(gradient)
        
        cost = np.mean(error ** 2)
        cost_history.append(cost)
        
        if np.isnan(cost) or np.isinf(cost):
            print("Numerical instability encountered. Exiting.")
            break
        
        if iteration % 10000 == 0:
            print(f"Iteration {iteration}, Cost {cost}, Gradient Magnitude {gradient_magnitude}")

        if gradient_magnitude < tolerance:
            print("Tolerance reached. Exiting.")
            break

        iteration += 1

    weights[1:] = weights[1:] / (X_max - X_min)
    weights[0] = weights[0] - np.sum(weights[1:] * X_min)

    return weights, cost_history

# create initial guess
cheapest_house = train_set.loc[train_set['price'].idxmin()]
most_expensive_house = train_set.loc[train_set['price'].idxmax()]
smallest_house = train_set.loc[train_set['sqft_living'].idxmin()]
largest_house = train_set.loc[train_set['sqft_living'].idxmax()]

midpoint_price = (cheapest_house['price'] + most_expensive_house['price']) / 2
midpoint_sqft = (smallest_house['sqft_living'] + largest_house['sqft_living']) / 2

price_slope = (most_expensive_house['price'] - cheapest_house['price']) / (most_expensive_house['sqft_living'] - cheapest_house['sqft_living'])
sqft_slope = (largest_house['price'] - smallest_house['price']) / (largest_house['sqft_living'] - smallest_house['sqft_living'])
slope_l = (price_slope + sqft_slope) / 2

intercept_l1 = most_expensive_house['price'] - price_slope * most_expensive_house['sqft_living']
intercept_l2 = largest_house['price'] - sqft_slope * largest_house['sqft_living']
intercept_l = (intercept_l1 + intercept_l2) / 2

print('Gradient Descent Model:')
print(f'Intercept (initial guess): {intercept_l:.2f}')
print(f'Slope (initial guess): {slope_l:.2f}')

# fit the model
initial_weights = [intercept_l, slope_l]
alpha = 5e-3
tolerance = 1e-7
max_iterations = 5e27
weights, cost_history = simple_linear_regression3(X, y, initial_weights, alpha, tolerance, max_iterations)

final_cost = cost_history[-1]

print(f"Final Weight Vector: {weights}")
print(f"Final Cost: {final_cost}")
print(f"Alpha: {alpha}, Tolerance: {tolerance}, Iterations: {len(cost_history)}")

# plt.plot(cost_history)
# plt.xlabel('Iteration')
# plt.ylabel('Cost')
# plt.title('Cost Function Over Iterations')
# plt.show()


# evaluate model quality
average_price = test_set['price'].mean()

# create the design matrix for the test set, adding a column of ones for the intercept
X_test = np.column_stack((np.ones(len(test_set)), test_set['sqft_living']))

# predict the prices using the optimal model weights
predictions_optimal = predict(X_test, w2)

# calculate the unsigned error and RMSE
unsigned_error = np.abs(predictions_optimal - test_set['price']).mean()
rmse = np.sqrt(np.mean((predictions_optimal - test_set['price']) ** 2))
test_set_optimal_cost = np.mean((predictions_optimal - test_set['price']) ** 2)

# print the results
print(f"Average Price: {average_price}")
print(f"Average Unsigned Error: {unsigned_error}")
print(f"RMSE: {rmse}")
print(f"MSE: {optimal_cost}")
percentage_diff = ((final_cost - optimal_cost) / optimal_cost) * 100
print(f"Percentage Difference in the Closed Form Model's Cost and the Gradient Descent Model's Cost: {percentage_diff}")

# plt.figure(figsize=(10, 6))

plt.scatter(test_set['sqft_living'], test_set['price'], color='black', label='Actual Price')
plt.scatter(test_set['sqft_living'], predictions_optimal, color='red', label='Optimal Model Predictions')

X_test = np.column_stack((np.ones(len(test_set)), test_set['sqft_living']))
predictions_gd = predict(X_test, weights)
test_set_gd_cost = np.mean((predictions_gd - test_set['price']) ** 2)
percentage_diff_test = ((test_set_gd_cost - test_set_optimal_cost) / test_set_optimal_cost) * 100
print('Percentage difference between the optimal model and the gradient descent model is: ', percentage_diff_test)
plt.scatter(test_set['sqft_living'], predictions_gd, color='green', label='Gradient Descent Predictions')

plt.xlabel('Square Feet Living')
plt.ylabel('Price')
plt.title('House Prices Prediction')
plt.legend()
plt.show()
