import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

np.set_printoptions(precision=2,suppress=True)

from platform import python_version

print('Using Python',python_version(), 'and NumPy version', np.__version__)

# use WIDER CANVAS:
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

sales = pd.read_csv('./data/home_data')
print('Number of house sales:', len(sales), '\nAvailable features:\n',sales.columns)

train_data = sales.sample(frac=0.8, random_state=0)
test_data = sales.drop(train_data.index)
print(f'Split data into {len(train_data)} training samples and {len(test_data)} testing samples')

## multiple regression

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

def multiple_regression_closed(X, y):
    XtX = np.dot(X.T, X)
    XtX_inv = np.linalg.inv(XtX)
    XtX_inv_Xt = np.dot(XtX_inv, X.T)
    weights = np.dot(XtX_inv_Xt, y)

    return weights

# compute weights using the training set on the closed model
X_train_closed, y_train_closed = pandas2numpy(train_data, ['sqft_living'], 'price')
weights_closed_model = multiple_regression_closed(X_train_closed, y_train_closed)
print("Computed model weights:", weights_closed_model)

#include sqft_living15 as a feature
X, y = pandas2numpy(train_data, ['sqft_living', 'sqft_living15'], 'price')
weights_closed_model_2 = multiple_regression_closed(X, y)
print("Computed model weights:", weights_closed_model_2)

# compute predictions on the test data using the closed-form solution
test_X, test_y = pandas2numpy(test_data, ['sqft_living', 'sqft_living15'], 'price')
test_predictions = predict_target(test_X, weights_closed_model_2)

# predict the price of the first house using the closed-form solution
first_house_predicted_price = test_predictions[0]
first_house_predicted_price_rounded = round(first_house_predicted_price)
print(f"The predicted price for the 1st house is: ${first_house_predicted_price_rounded}")


def MSE(y,y_hat):
    if len(y) != len(y_hat):
        raise ValueError("The length of y and y_hat must be the same.")
    mse = np.mean((y - y_hat) ** 2)
    return mse

# compute the MSE of the closed-form solution on the test data
mse = MSE(test_y, test_predictions)
rmse = sqrt(mse)
print("Root Mean Squared Error (RMSE) on Test Data:", rmse)


## polynomial regression

# compute the RMSE of the closed-form solution based only on `'sqft_living'
test_predictions_simple_model = predict_target(X_train_closed, weights_closed_model)
mse4 = MSE(y_train_closed, test_predictions_simple_model)
rmse4 = sqrt(mse4)
print("Root Mean Squared Error (RMSE) on Test Data:", rmse4)


# create a NumPy array consisting of a column of ones followed by all the integer powers of a column `feature` 
    # up to a specific degree
def polynomial_matrix(frame, feature, degree, target):
    poly_frame = pd.DataFrame()
    if degree < 1:
        raise ValueError("Degree must be at least 1")
    for power in range(1, degree + 1):
        column_name = f'power_{power}'
        poly_frame[column_name] = frame[feature] ** power
    y = frame[target].values
    return poly_frame, y


# This time we scale areas to 1000th of sqft to attenuate numerical precision errors.
# We also sort the data by area and resolve ties by price.
sales['sqft_living']=sales['sqft_living']/1000.
sales = sales.sort_values(['sqft_living', 'price'])


# fit a 1-degree plot:
X11, y11 = polynomial_matrix(sales, 'sqft_living', 1, 'price')
model11_weights = multiple_regression_closed(X11, y11)
print("Computed model weights:", model11_weights)
predicted_prices = predict_target(X11, model11_weights)

plt.figure(figsize=(10, 6))
plt.scatter(X11['power_1'], y11, color='blue', label='Actual Price', alpha=0.5)
plt.plot(X11['power_1'], predicted_prices, color='red', label='Predicted Price')
plt.xlabel('Living Area (in 1000 sqft)')
plt.ylabel('Price')
plt.title('Degree-1 Polynomial Regression - Price vs Living Area')
plt.legend()
plt.show()

# fit a 2-degree plot:
X_2degree_model, y_2degree_model = polynomial_matrix(sales, 'sqft_living', 2, 'price')
degree2_model_weights = multiple_regression_closed(X_2degree_model, y_2degree_model)
print("Learned coefficients for the degree-2 model:", degree2_model_weights)
sales_sorted = sales.sort_values(['sqft_living'])
X_2degree_model_sorted, _ = polynomial_matrix(sales_sorted, 'sqft_living', 2, 'price')
predicted_prices_2degree_model = predict_target(X_2degree_model_sorted, degree2_model_weights)

plt.figure(figsize=(10, 6))
plt.scatter(sales['sqft_living'], sales['price'], color='blue', label='Actual Price', alpha=0.5)
plt.plot(sales_sorted['sqft_living'], predicted_prices_2degree_model, color='red', label='Degree-2 Model')
plt.xlabel('Living Area (in 1000 sqft)')
plt.ylabel('Price')
plt.title('Degree-2 Polynomial Regression - Price vs Living Area')
plt.legend()
plt.show()

# fit a 3-degree plot:
X_3degree_model, y_3degree_model = polynomial_matrix(sales, 'sqft_living', 3, 'price')
degree3_model_weights = multiple_regression_closed(X_3degree_model, y_3degree_model)
print("Learned coefficients for the degree-3 model:", degree3_model_weights)
sales_sorted = sales.sort_values(['sqft_living'])
X_3degree_model_sorted, _ = polynomial_matrix(sales_sorted, 'sqft_living', 3, 'price')
predicted_prices_3degree = predict_target(X_3degree_model_sorted, degree3_model_weights)

plt.figure(figsize=(10, 6))
plt.scatter(sales['sqft_living'], sales['price'], color='blue', label='Actual Price', alpha=0.5)
plt.plot(sales_sorted['sqft_living'], predicted_prices_3degree, color='red', label='Degree-3 Model')
plt.xlabel('Living Area (in 1000 sqft)')
plt.ylabel('Price')
plt.title('Degree-3 Polynomial Regression - Price vs Living Area')
plt.legend()
plt.show()


# fit a 10-degree plot:
X_10degree_model, y_10degree_model = polynomial_matrix(sales, 'sqft_living', 10, 'price')
degree10_model_weights = multiple_regression_closed(X_10degree_model, y_10degree_model)
print("Learned coefficients for the degree-10 model:", degree10_model_weights)
sales_sorted = sales.sort_values(['sqft_living'])
X_10degree_model_sorted, _ = polynomial_matrix(sales_sorted, 'sqft_living', 10, 'price')
predicted_prices_10degree_model = predict_target(X_10degree_model_sorted, degree10_model_weights)

plt.figure(figsize=(10, 6))
plt.scatter(sales['sqft_living'], sales['price'], color='blue', label='Actual Price', alpha=0.5)
plt.plot(sales_sorted['sqft_living'], predicted_prices_10degree_model, color='red', label='Degree-3 Model')
plt.xlabel('Living Area (in 1000 sqft)')
plt.ylabel('Price')
plt.title('Degree-10 Polynomial Regression - Price vs Living Area')
plt.legend()
plt.show()



## exploring variance using different sets:
half_1, half_2 = sales.sample(frac=0.5, random_state=0), sales.drop(sales.sample(frac=0.5, random_state=0).index)
set_1, set_2 = half_1.sample(frac=0.5, random_state=0), half_1.drop(half_1.sample(frac=0.5, random_state=0).index)
set_3, set_4 = half_2.sample(frac=0.5, random_state=0), half_2.drop(half_2.sample(frac=0.5, random_state=0).index)

# fit a 10-degree model on each of the sets
def plot_poly_fit(data_set, feature, degree, target, i):
    poly_data, target_data = polynomial_matrix(data_set, feature, degree, target)
    model_weights = multiple_regression_closed(poly_data, target_data)
    print(f"Dataset number: {i+1}")
    print("Model coefficients:", model_weights)
    poly_data_sorted = poly_data.sort_values(by='power_1')
    predictions = predict_target(poly_data_sorted, model_weights)
    plt.plot(poly_data_sorted['power_1'], predictions, color='red', label=f'Set {i+1}')
    plt.scatter(data_set[feature], data_set[target], alpha=0.5, label='Actual Data')

for i, data_set in enumerate([set_1, set_2, set_3, set_4]):
    plt.figure(figsize=(10, 6))
    plot_poly_fit(data_set, 'sqft_living', 10, 'price', i)
    plt.xlabel('sqft_living (in 1000 sqft)')
    plt.ylabel('Price')
    plt.title(f'10th Degree Polynomial Fit - Set {i+1}')
    plt.legend()
    plt.show()



## select an appropriate degree:

# split the data
training_and_validation, testing = train_test_split(sales, test_size=0.1, random_state=3)
training, validation = train_test_split(training_and_validation, test_size=0.5, random_state=3)

# instantiate
best_degree = 1
lowest_mse = float('inf')
best_weights = None
best_weights_dict = {}

for degree in range(1, 11):
    poly_frame_train, target_train = polynomial_matrix(training, 'sqft_living', degree, 'price')

    poly_frame_train['intercept'] = 1
    columns_train = ['intercept'] + [col for col in poly_frame_train if col != 'intercept']
    poly_frame_train = poly_frame_train[columns_train]
    
    model_weights = multiple_regression_closed(poly_frame_train.values, target_train)
    best_weights_dict[degree] = model_weights[1:]

    poly_frame_val, target_val = polynomial_matrix(validation, 'sqft_living', degree, 'price')
    
    # add a column of ones to poly_frame_val for the intercept
    poly_frame_val['intercept'] = 1
    
    columns_val = ['intercept'] + [col for col in poly_frame_val if col != 'intercept']
    poly_frame_val = poly_frame_val[columns_val]
    
    predictions_val = predict_target(poly_frame_val.values, model_weights)
    
    mse = MSE(target_val, predictions_val)
    print(f"MSE for degree {degree}: {mse}")
    
    if mse < lowest_mse:
        lowest_mse = mse
        best_degree = degree
        best_weights = model_weights

print(f"The degree {best_degree} polynomial had the lowest MSE on validation data.")

# plot the magnitude of model coefficients vs the polynomial degree
degrees = list(range(1, 11))
l2_norms = []
for degree in degrees:
    l2_norm = np.linalg.norm(best_weights_dict[degree])
    l2_norms.append(l2_norm)
plt.figure(figsize=(10, 6))
plt.plot(degrees, l2_norms, marker='o')
plt.title('Magnitude of Model Coefficients vs. Polynomial Degree')
plt.xlabel('Polynomial Degree')
plt.ylabel('L2 Norm of Coefficients')
plt.grid(True)
plt.show()

# find RMSE of test data, using the validation data
X_train_val, y_train_val = polynomial_matrix(validation, 'sqft_living', 3, 'price')
degree3_model_weights = multiple_regression_closed(X_train_val, y_train_val)
X_test, y_test = polynomial_matrix(testing, 'sqft_living', 3, 'price')
predicted_prices_test = predict_target(X_test, degree3_model_weights)
rmse_test = np.sqrt(MSE(y_test, predicted_prices_test))
print(f"The RMSE of the test data for the degree-3 model is: {rmse_test}")



## improve the MSE by 4x by creating a new model using regression:
target = 'price'
numeric_features = ['sqft_living', 'sqft_lot','grade', 'lat', 'long', 'waterfront', 'view', 'condition', 'yr_built', 'yr_renovated', 'sqft_above', 'sqft_basement']
categorical_features = ['zipcode']

# preprocessing pipelines for numeric and categorical data: numeric - normalization, categorical - one hot
numeric_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler()) 
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore')) 
])

# bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# instantiate the Gradient Boosting model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=3, min_samples_leaf=1, min_samples_split=2))])

# fit the model to the data
pipeline.fit(validation.drop(columns=target), validation[target])

# Predict on the test data
predicted_prices_test_pipeline = pipeline.predict(validation.drop(columns=target))

# Calculate RMSE using testing target instead of validation target
rmse_test_pipeline = np.sqrt(MSE(validation[target], predicted_prices_test_pipeline))
print(f"The RMSE using the normalized Gradient Boosting model is: {rmse_test_pipeline}")

# Calculate the improvement factor based on the initial RMSE test value
improvement_factor_pipeline = rmse_test / rmse_test_pipeline
print(f"The improvement factor using the normalized Gradient Boosting model is: {improvement_factor_pipeline}")