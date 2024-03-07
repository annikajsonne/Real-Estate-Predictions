import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import warnings 
from sklearn.metrics import mean_squared_error

np.set_printoptions(precision=2,suppress=True)

sales = pd.read_csv('data/home_data.gz')
print(f'Input features:\n {sales.columns}')

# create new features:
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


# create a lasso model using all the features and a 1E6 L1 penalty
y = sales['price'] #target
X = sales[all_features]
lasso_model = Lasso(alpha=1E6, max_iter=10000)
lasso_model.fit(X, y)

# find the non-zero features and intercept
features_and_coefficients = list(zip(all_features, lasso_model.coef_))
print("Coefficients:", lasso_model.coef_)
non_zero_features = [feature for feature, coefficient in features_and_coefficients if coefficient != 0]
print("Non-zero features:")
print(non_zero_features)
print("Intercept:", lasso_model.intercept_)


## selecting a L1 penalty

# split the data
train_validation_data, test_data = train_test_split(sales, test_size=0.1, random_state=0)
train_data, validation_data = train_test_split(train_validation_data, test_size=0.5, random_state=0)
print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(validation_data)}")
print(f"Test set size: {len(test_data)}")

# fit several models with different L1 penalties
    # sklearn will flag bad choices for the l1 penalty with a warning. Let's just ignore those
warnings.filterwarnings("ignore")

X_train = train_data[all_features]
y_train = train_data['price']
X_validation = validation_data[all_features]
y_validation = validation_data['price']

l1_penalties = np.logspace(2, 9, num=15)
mse_values = []
non_zero_weights = []

for l1_penalty in l1_penalties:
    model = Lasso(alpha=l1_penalty, max_iter=10000)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_validation)
    
    mse = mean_squared_error(y_validation, predictions)
    mse_values.append(mse)
    
    non_zero_count = np.count_nonzero(model.coef_) + np.count_nonzero([model.intercept_])
    non_zero_weights.append(non_zero_count)

for i, l1_penalty in enumerate(l1_penalties):
    print(f"L1 penalty: {l1_penalty}")
    print(f"MSE: {mse_values[i]}")
    print(f"Number of non-zero weights: {non_zero_weights[i]}\n")


# find L1 penalty with the minimum error
mse_values_array = np.array(mse_values)
min_mse_index = mse_values_array.argmin()
optimal_l1_penalty = l1_penalties[min_mse_index]
print(f"The l1_penalty that yielded the minimum MSE is: {optimal_l1_penalty}")
print(f"Minimum MSE: {mse_values[min_mse_index]}")
print(f"Number of non-zero weights at this penalty: {non_zero_weights[min_mse_index]}")


# fit 20 log-spaced penalty values between 1E5 and 1E7 inclusive, and find the l1 penalties and number of non zero weights
non_zero_counts = []

l1_penalty_values = np.logspace(5, 7, num=20)

for l1_penalty in l1_penalty_values:
    model = Lasso(alpha=l1_penalty, max_iter=10000)
    model.fit(X_train, y_train)

    non_zero_count = np.count_nonzero(model.coef_) + np.count_nonzero([model.intercept_])
    non_zero_counts.append(non_zero_count)

for i, l1_penalty in enumerate(l1_penalty_values):
    print(f"L1 penalty: {l1_penalty}")
    print(f"Number of non-zero weights: {non_zero_counts[i]}\n")

# find the min and max l1 penalties that give the desired number of non-zero weights

# use 5 non zero weights
target_nonzeros = 5

l1_penalty_min = None
l1_penalty_max = None
for i in range(len(l1_penalty_values)):
    if non_zero_counts[i] > target_nonzeros:
        l1_penalty_min = l1_penalty_values[i]

for i in range(len(l1_penalty_values) - 1, -1, -1):
    if non_zero_counts[i] < target_nonzeros:
        l1_penalty_max = l1_penalty_values[i]
        break

print(f"l1_penalty_min: {l1_penalty_min}")
print(f"l1_penalty_max: {l1_penalty_max}")

# find l1 penalties within the range
l1_penalty_values = np.linspace(l1_penalty_min,l1_penalty_max,20)

# find the model with the lowest MSE on the validation set
best_l1_penalty = None
lowest_mse = np.inf
best_non_zero_count = None
l1_penalty_values = np.linspace(l1_penalty_min, l1_penalty_max, 20)

for l1_penalty in l1_penalty_values:
    model = Lasso(alpha=l1_penalty, max_iter=10000)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_validation)
    mse = mean_squared_error(y_validation, predictions)
    
    non_zero_count = np.count_nonzero(model.coef_) + np.count_nonzero([model.intercept_])
    
    if non_zero_count == target_nonzeros and mse < lowest_mse:
        best_l1_penalty = l1_penalty
        lowest_mse = mse
        best_non_zero_count = non_zero_count

print(f"Best L1 penalty: {best_l1_penalty}")
print(f"Lowest MSE on the validation set: {lowest_mse}")
print(f"Number of non-zero weights: {best_non_zero_count}")


print(f"The l1_penalty that has the lowest RSS on validation and sparsity equal to target_nonzeros is {best_l1_penalty}")
optimal_model = Lasso(alpha=best_l1_penalty, max_iter=10000)
optimal_model.fit(X_train, y_train)

# extract the feature names and their corresponding coefficients
features_and_coefficients = list(zip(all_features, optimal_model.coef_))

# determine which features have non-zero coefficients
non_zero_features = [feature for feature, coefficient in features_and_coefficients if coefficient != 0]
print("Features with non-zero coefficients:")
print(non_zero_features)