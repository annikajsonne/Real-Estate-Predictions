import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

import warnings 
warnings.filterwarnings('ignore')

np.set_printoptions(precision=2,suppress=True)

# load in house data
sales = pd.read_csv('data/home_data.gz')
print(f'Input features:\n {sales.columns}')

# create new features
sales['sqft_living_sqrt'] = np.sqrt(sales['sqft_living'])
sales['sqft_lot_sqrt'] = np.sqrt(sales['sqft_lot'])
sales['bedrooms_square'] = sales['bedrooms'] ** 2
sales['floors_square'] = sales['floors'] ** 2
print(f'Updated features:\n {sales.columns}')

all_features = ['bedrooms', 'bedrooms_square',
                'bathrooms',
                'sqft_living', 'sqft_living_sqrt',
                'sqft_lot', 'sqft_lot_sqrt',
                'floors', 'floors_square',
                'waterfront', 'view', 'condition', 'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built', 'yr_renovated']


# fit model with all the available features

X = sales[all_features]
y = sales['price'] # target
lasso_model = Lasso(alpha=1E6, max_iter=10000)
lasso_model.fit(X, y)

coefficients = list(zip(all_features, lasso_model.coef_))
#print(f"Coefficients: {lasso_model.coef_}")

# find the nonzero features
non_zero = [feature for feature, coefficient in coefficients if coefficient != 0]
print(f"Non-zero features: {non_zero}")
print(f"Intercept term: {lasso_model.intercept_}")

# split data into training, testing and validation sets
train_validation_data, test_data = train_test_split(sales, test_size=0.1, random_state=0)
train_data, validation_data = train_test_split(train_validation_data, test_size=0.5, random_state=0)
print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(validation_data)}")
print(f"Test set size: {len(test_data)}")

# fit several models with different L1 penalties
l1_penalties = np.logspace(2, 9, num=15)

mse_values = []
non_zero_weights = []

for l1_penalty in l1_penalties:
    lasso_model = Lasso(alpha=l1_penalty, max_iter=10000)
    lasso_model.fit(train_data[all_features], train_data['price'])
    predictions = lasso_model.predict(validation_data[all_features])
    mse = mean_squared_error(validation_data['price'], predictions)
    mse_values.append(mse)
    non_zero_count = np.sum(lasso_model.coef_ != 0) + (lasso_model.intercept_ != 0)
    non_zero_weights.append(non_zero_count)
    print(f"L1 Penalty: {l1_penalty}, MSE: {mse}, Non-zero weights: {non_zero_count}")

# find the model that minimizes the MSE
min_mse = min(mse_values)
min_mse_index = mse_values.index(min_mse)
optimal_l1_penalty = l1_penalties[min_mse_index]

print(f"The l1_penalty that yielded the minimum MSE is: {optimal_l1_penalty}")
print(f"Minimum MSE: {mse_values[min_mse_index]}")
print(f"Number of non-zero weights at this penalty: {non_zero_weights[min_mse_index]}")

# limit the number of nonzero weights
target_nonzeros = 5

# fit 20 log-spaced penalty values between 1E5 and 1E7 inclusive
l1_penalties = np.logspace(5, 7, num=20)
non_zero_counts = []
for l1_penalty in l1_penalties:
    lasso_model = Lasso(alpha=l1_penalty, max_iter=10000)
    lasso_model.fit(train_data[all_features], train_data['price'])
    non_zero_count = np.count_nonzero(lasso_model.coef_) + (lasso_model.intercept_ != 0)
    non_zero_counts.append(non_zero_count)
    print(f"L1 Penalty: {l1_penalty}")
    print(f"Non-zero weights: {non_zero_count}")

# find the minimum and maximum L1 penalties
l1_penalty_min = None
l1_penalty_max = None

for i in range(len(l1_penalties)-1, -1, -1):
    if non_zero_counts[i] > target_nonzeros:
        l1_penalty_min = l1_penalties[i]
        break

for i in range(len(l1_penalties)):
    if non_zero_counts[i] < target_nonzeros:
        l1_penalty_max = l1_penalties[i]
        break 

print(f"l1_penalty_min: {l1_penalty_min}")
print(f"l1_penalty_max: {l1_penalty_max}")


# find the model with the lowest mse on the validation data
l1_penalty_values = np.linspace(l1_penalty_min,l1_penalty_max,20)
best_l1_penalty = None
lowest_mse = np.inf
best_non_zero_count = None

l1_penalty_values = np.linspace(l1_penalty_min, l1_penalty_max, 20)

for l1_penalty in l1_penalty_values:
    model = Lasso(alpha=l1_penalty, max_iter=10000)
    model.fit(train_data[all_features], train_data['price'])
    
    predictions = model.predict(validation_data[all_features])
    mse = mean_squared_error(validation_data['price'], predictions)
    
    non_zero_count = np.count_nonzero(model.coef_) + np.count_nonzero([model.intercept_])
    
    if non_zero_count == target_nonzeros and mse < lowest_mse:
        best_l1_penalty = l1_penalty
        lowest_mse = mse
        best_non_zero_count = non_zero_count

print(f"Best L1 penalty: {best_l1_penalty}")
print(f"Lowest MSE on the validation set: {lowest_mse}")
print(f"Number of non-zero weights: {best_non_zero_count}")

print(f"The l1_penalty that has the lowest RSS on validation and sparsity equal to target_nonzeros is {best_l1_penalty}")

# find the optimal model and the nonzero coefficients
optimal_model = Lasso(alpha=best_l1_penalty, max_iter=10000)
optimal_model.fit(train_data[all_features], train_data['price'])

features_and_coefficients = list(zip(all_features, optimal_model.coef_))

non_zero_features = [feature for feature, coefficient in features_and_coefficients if coefficient != 0]

print("Features with non-zero coefficients:")
print(non_zero_features)


def pandas2numpy(dataframe, features, target):
    dataframe['constant'] = 1
    columns = ['constant'] + features
    X = dataframe[columns].to_numpy()
    y = dataframe[target].to_numpy()
    return X, y

def predict_target(design_matrix, model):
    # assume feature_matrix is a NumPy matrix containing the features as columns and weights is a corresponding NumPy array
    predictions = design_matrix.dot(model)
    return predictions

X = np.array([[3.,5.,8.],[4.,12.,15.]])
print(X)

two_norms = np.linalg.norm(X, axis=0)

X_normalized = X / two_norms

print("2-norms of each column:", two_norms)
print("Normalized X:")
print(X_normalized)

def normalize_features(feature_matrix):
    """
    normalize the features in the feature matrix

    Parameters:
    feature_matrix (numpy matrix): The feature matrix to be normalized.

    Returns:
    tuple: A tuple containing the normalized feature matrix and the norms of the original features.
    """
    # Compute the 2-norms of each column (feature)
    norms = np.linalg.norm(feature_matrix, axis=0)
    
    # Normalize the feature matrix
    normalized_features = feature_matrix / norms
    
    return normalized_features, norms

features, norms = normalize_features(np.array([[3.,6.,9.],[4.,8.,12.]]))
print(features)
# should print
# [[ 0.6  0.6  0.6]
#  [ 0.8  0.8  0.8]]
print(norms)
# should print
# [5.  10.  15.]

simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
simple_feature_matrix, output = pandas2numpy(sales, simple_features, my_output)

simple_feature_matrix, norms = normalize_features(simple_feature_matrix)

weights = np.array([1., 4., 1.])

prediction = predict_target(simple_feature_matrix,weights)

r_j_values = np.zeros(len(weights))

for j in range(len(weights)):
    # Start by making a prediction using all features
    feature_contribution = weights[j] * simple_feature_matrix[:, j]
    residual = output - (prediction - feature_contribution)
    
    r_j_values[j] = simple_feature_matrix[:, j].dot(residual)

l1_penalty_lower_bound = 2 * abs(r_j_values[2])
l1_penalty_upper_bound = 2 * abs(r_j_values[1])

(l1_penalty_lower_bound, l1_penalty_upper_bound)

l1_penalty_threshold = 2 * max(abs(r_j_values[1]), abs(r_j_values[2]))

def lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty):
    """
    Computes the new weight for a single feature in LASSO regression.
    
    Parameters:
    i (int): The index of the feature to be optimized.
    feature_matrix (numpy.ndarray): The feature matrix.
    output (numpy.ndarray): The array of output values.
    weights (numpy.ndarray): The current set of weights.
    l1_penalty (float): The L1 penalty term.
    
    Returns:
    float: The updated weight for feature i.
    """
    prediction = predict_target(feature_matrix, weights)
    
    ro_i = feature_matrix[:, i].dot(output - prediction + weights[i] * feature_matrix[:, i])
    
    if i == 0: 
        new_weight_i = ro_i
    else:
        if ro_i < -l1_penalty / 2.0:
            new_weight_i = ro_i + l1_penalty / 2
        elif ro_i > l1_penalty / 2.0:
            new_weight_i = ro_i - l1_penalty / 2
        else:
            new_weight_i = 0.0
    
    return new_weight_i

def lasso_cyclical_coordinate_descent(feature_matrix, output, initial_weights, l1_penalty, tolerance):
    """
    Optimizes the weights for a LASSO regression model using cyclical coordinate descent.
    
    Parameters:
    feature_matrix (numpy.ndarray): The feature matrix.
    output (numpy.ndarray): The array of output values.
    initial_weights (numpy.ndarray): The initial set of weights.
    l1_penalty (float): The L1 penalty term.
    tolerance (float): The convergence tolerance.
    
    Returns:
    numpy.ndarray: The optimized weights.
    """
    weights = np.array(initial_weights)
    change = np.zeros(len(weights))  
    converged = False
    
    while not converged:
        max_change = 0 
        for i in range(len(weights)):
            old_weight_i = weights[i]
            
            weights[i] = lasso_coordinate_descent_step(i, feature_matrix, output, weights, l1_penalty)
 
            change[i] = np.abs(old_weight_i - weights[i])
            
            max_change = max(max_change, change[i])
        
        if max_change < tolerance:
            converged = True
    
    return weights

simple_features = ['sqft_living', 'bedrooms']
my_output = 'price'
initial_weights = np.zeros(3)
l1_penalty = 1e7
tolerance = 1.0

simple_feature_matrix, output = pandas2numpy(sales, simple_features, my_output)
normalized_simple_feature_matrix, norms = normalize_features(simple_feature_matrix)

weights = lasso_cyclical_coordinate_descent(normalized_simple_feature_matrix, output, 
                                            initial_weights, l1_penalty, tolerance)

predictions = predict_target(normalized_simple_feature_matrix, weights)
mse = mean_squared_error(output, predictions)

feature_names = ['constant'] + simple_features
zero_weight_features = [feature_names[i] for i in range(len(weights)) if weights[i] == 0]

print("MSE of the learned model on normalized dataset:", mse)
print("Features with zero weight:", zero_weight_features)

train_data = sales.sample(frac=0.8, random_state=0)
test_data = sales.drop(train_data.index)

all_features = ['bedrooms',
                'bathrooms',
                'sqft_living',
                'sqft_lot',
                'floors',
                'waterfront', 
                'view', 
                'condition', 
                'grade',
                'sqft_above',
                'sqft_basement',
                'yr_built', 
                'yr_renovated']

my_output = 'price'
initial_weights = np.zeros(1+len(all_features))
l1_penalty = 1e7
tolerance = 1.0
(all_feature_matrix, output) = pandas2numpy(train_data, all_features, my_output)
(normalized_all_feature_matrix, all_norms) = normalize_features(all_feature_matrix)

weights1e7 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, output,
                                               initial_weights, l1_penalty, tolerance)

all_feature_names = ['constant'] + all_features 

non_zero_weight_features = [feature for i, feature in enumerate(all_feature_names) if weights1e7[i] != 0]

print("Features with non-zero weights:", non_zero_weight_features)

l1_penalty = 1e8
initial_weights = np.zeros(1+len(all_features))

weights1e8 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, output,
                                               initial_weights, l1_penalty, tolerance)

non_zero_weight_features1e8 = [feature for i, feature in enumerate(all_feature_names) if weights1e8[i] != 0]

print("Features with non-zero weights:", non_zero_weight_features1e8)

l1_penalty = 1e4
initial_weights = np.zeros(1+len(all_features))

tolerance = 5e5

weights1e4 = lasso_cyclical_coordinate_descent(normalized_all_feature_matrix, output,
                                               initial_weights, l1_penalty, tolerance)

non_zero_weight_features1e4 = [feature for i, feature in enumerate(all_feature_names) if weights1e4[i] != 0]

print("Features with non-zero weights:", non_zero_weight_features1e4)

normalized_weights1e4 = weights1e4 / all_norms

normalized_weights1e7 = weights1e7 / all_norms

normalized_weights1e8 = weights1e8 / all_norms

print("Normalized weight for feature 3 with l1_penalty=1e7:", normalized_weights1e7[3])

test_feature_matrix, test_output = pandas2numpy(test_data, all_features, 'price')

predictions1e4 = predict_target(test_feature_matrix, normalized_weights1e4)
predictions1e7 = predict_target(test_feature_matrix, normalized_weights1e7)
predictions1e8 = predict_target(test_feature_matrix, normalized_weights1e8)

mse1e4 = mean_squared_error(test_output, predictions1e4)
mse1e7 = mean_squared_error(test_output, predictions1e7)
mse1e8 = mean_squared_error(test_output, predictions1e8)

print("MSE with l1_penalty=1e4:", mse1e4)
print("MSE with l1_penalty=1e7:", mse1e7)
print("MSE with l1_penalty=1e8:", mse1e8)

best_model = min(mse1e4, mse1e7, mse1e8)
if best_model == mse1e4:
    print("Best model: l1_penalty=1e4")
elif best_model == mse1e7:
    print("Best model: l1_penalty=1e7")
else:
    print("Best model: l1_penalty=1e8")