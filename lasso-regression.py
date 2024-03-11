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