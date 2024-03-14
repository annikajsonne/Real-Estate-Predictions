# Real Estate Predictions Using Regression

## Simple Linear Regression
*This Python script demonstrates simple linear regression using both closed-form solution and gradient descent optimization techniques. It provides a comprehensive example of implementing linear regression, evaluating model performance, and visualizing results.*

### Code Overview
#### Purpose
The purpose of this script is to showcase two methods for fitting a linear regression model to a dataset:

1. Closed-form Solution: Computes the optimal values of the slope and intercept using matrix operations.
2. Gradient Descent: Optimizes the model parameters iteratively to minimize the cost function.

#### Features
- Loads house sales data from a GZ file.
- Plots histograms for selected features to visualize data distributions.
- Computes linear regression models using both closed-form solution and Scikit-learn's Linear Regression.
- Implements gradient descent optimization for linear regression.
- Evaluates model quality using metrics such as mean squared error (MSE) and root mean squared error (RMSE).
- Visualizes the results by plotting actual prices against predicted prices using both optimal and gradient descent models.

### Usage
1. Data Preparation: Ensure you have the dataset ready in GZ format. The script expects a file named `home_data.gz` which is found within the `data` directory.
2. Running the Script: Execute the provided Python script (`simple-linear-regression.py`) in your preferred environment.
3. Interpreting Results: Analyze the generated plots and printed metrics to understand the performance of each regression method.

### Dependencies
Make sure you have the following Python libraries installed:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Multiple & Polynomial Regression 
*This Python script demonstrates polynomial regression and gradient boosting techniques for predicting house prices based on various features. It provides an extensive example of preprocessing, model fitting, evaluation, and comparison.*

### Code Overview
#### Purpose
The purpose of this script is to showcase the following machine learning tasks:

1. Multiple Regression: Predict house prices using multiple features such as square footage, location, and condition.
2. Polynomial Regression: Fit polynomial models of varying degrees to explore complex relationships between features and house prices.
3. Model Selection: Determine the appropriate degree for polynomial regression using validation data.
4. Gradient Boosting: Improve prediction accuracy using a gradient boosting regressor with feature preprocessing.

#### Features
- Loads house sales data from a GZ file and performs train-test split.
- Implements multiple regression using closed-form solution for feature selection.
- Explores polynomial regression with degrees ranging from 1 to 10 to capture non-linear relationships.
- Selects the best polynomial degree based on validation set performance.
- Utilizes gradient boosting with preprocessing pipelines for improved prediction accuracy.
- Evaluates models using root mean squared error (RMSE) and compares their performance.

### Usage
1. Data Preparation: Ensure you have the dataset ready in GZ format. The script expects a file named `home_data.gz` which is found within the `data` directory.
2. Running the Script: Execute the provided Python script (`multiple-polynomial-regression.py`) in your preferred environment.
3. Interpreting Results: Analyze the generated plots and printed metrics to understand the performance of each regression method.

### Dependencies
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Lasso Regression
*This Python script demonstrates the use of Lasso regression for predicting house prices. It includes feature engineering, model tuning using cross-validation, and evaluating model performance.*

### Code Overview
#### Purpose
The purpose of this script is to showcase the process of using Lasso regression to predict house prices. It covers the following steps:

1. Feature Engineering: Creating new features such as square roots and squared terms.
2. Model Training: Fitting Lasso regression models with different L1 penalties on training data.
3. Model Evaluation: Selecting the optimal L1 penalty based on validation set performance.
4. Feature Selection: Identifying the most important features with non-zero coefficients.
5. Model Testing: Evaluating the final model's performance on unseen test data.

#### Features
- Loads house sales data from the GZ file.
- Creates new features by transforming existing ones.
- Fits Lasso regression models with various L1 penalties and evaluates their performance using mean squared error (MSE).
- Selects the optimal L1 penalty based on the validation set's MSE.
- Identifies the most important features by examining non-zero coefficients.
- Tests the final model's performance on the test set.

### Usage
1. Data Preparation: Ensure you have the dataset ready in GZ format. The script expects a file named `home_data.gz` which is found within the `data` directory.
2. Running the Script: Execute the provided Python script (`lasso_regression.py`) in your preferred environment.
3. Interpreting Results: Analyze the printed outputs and model performance metrics to understand the effectiveness of Lasso regression for predicting house prices.

### Dependencies
- NumPy
- Pandas
- Scikit-learn

## Conclusion
In conclusion, this repository offers a comprehensive exploration of different regression techniques for predicting real estate prices. Starting with simple linear regression, the scripts demonstrate both closed-form solutions and gradient descent optimization, providing insights into model fitting, evaluation, and visualization. Moving on to multiple and polynomial regression, the focus shifts to capturing complex relationships between features and house prices, along with model selection and performance comparison using metrics like root mean squared error. Lastly, the incorporation of Lasso regression introduces feature engineering, model tuning via cross-validation, and feature selection to improve prediction accuracy. Overall, these scripts provide a valuable resource for anyone interested in real estate prediction using regression techniques, offering practical examples, usage guidelines, and insights into model performance.