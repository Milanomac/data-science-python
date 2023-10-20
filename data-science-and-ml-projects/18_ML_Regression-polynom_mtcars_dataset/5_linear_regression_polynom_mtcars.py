import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats as stats

matplotlib.style.use('ggplot')

# Load mtcars data set

mtcars = pd.read_csv('datasets/mtcars.xls')

# mtcars.plot(kind="scatter",
#             x = "wt",
#             y = "mpg",
#             figsize=(9,9),
#             color="black");

# plt.show()

from sklearn import linear_model

# Initialize model
regression_model = linear_model.LinearRegression()

# Train the model using the mtcars data
regression_model.fit(pd.DataFrame(mtcars['wt']),    # Predictors (allows only 2D arrays)
                     y = mtcars['mpg'])             # Target variable

# Check trained model y-intercept
print(regression_model.intercept_)

# Check trained model coefficients
print(f'As the weight of the car increases by 1, the mpg changes by {regression_model.coef_}')

score = regression_model.score(X = pd.DataFrame(mtcars["wt"]), 
                               y = mtcars["mpg"])

print(score)


train_prediction = regression_model.predict(X = pd.DataFrame(mtcars["wt"]))

# Actual - prediction = residuals
residuals = mtcars["mpg"] - train_prediction

print(residuals.describe())

# Plotting data

mtcars.plot(kind="scatter",
            x = "wt",
            y = "mpg",
            figsize=(9,9),
            color="black",
            xlim = (0,7));

# Drawing regression line on the plot

plt.plot(mtcars["wt"],
        train_prediction,
        color="blue")

# plt.show()
# =========================================Influence of the outlier====================
mtcars_subset = mtcars[["mpg","wt"]]

super_car = pd.DataFrame({"mpg":50,"wt":10}, index=["super"])

new_cars = pd.concat([mtcars_subset, super_car])

# Initialize model
regression_model = linear_model.LinearRegression()

# Train the model using the new_cars data
regression_model.fit(X = pd.DataFrame(new_cars["wt"]), 
                     y = new_cars["mpg"])

train_prediction2 = regression_model.predict(X = pd.DataFrame(new_cars["wt"]))

# Plot the new model
new_cars.plot(kind="scatter",
           x="wt",
           y="mpg",
           figsize=(9,9),
           color="black", xlim=(1,11), ylim=(10,52))

# Plot regression line
plt.plot(new_cars["wt"],     # Explanatory variable
         train_prediction2,  # Predicted values
         color="blue")

plt.show()
# =======================================================================================
# Q-Q plot to investigate the normality of the residuals
# Residual = actual value - prediction

plt.figure(figsize=(9,9))

stats.probplot(residuals, dist="norm", plot=plt)
# plt.show()

# RMSE - common evaluation metric for predictions involving real numbers.
# It is a square root of squared residuals

def RMSE(predicted, targets):
    """
    Computes root mean squared error of two numpy ndarrays
    
    Args:
        predicted: an ndarray of predictions
        targets: an ndarray of target values
    
    Returns:
        The root mean squared error as a float
    """
    return (np.sqrt(np.mean((targets-predicted)**2)))

print(RMSE(train_prediction, mtcars['mpg']))

from sklearn.metrics import mean_squared_error

RMSE = mean_squared_error(train_prediction, mtcars["mpg"])**0.5

# RMSE

# ==================================POLYNOMIAL REGRESSION===============================

# Initialize model
poly_model = linear_model.LinearRegression()

# Make a DataFrame of predictor variables
predictors = pd.DataFrame([mtcars["wt"],           # Include weight
# t
                           mtcars["wt"]**2]).T     # Include weight squared

# t squared

# Train the model using the new_cars data
poly_model.fit(X = predictors, 
               y = mtcars["mpg"])

# Check trained model y-intercept
print("Model intercept")
print(poly_model.intercept_)

# Check trained model coefficients (scaling factor given to "wt")
print("Model Coefficients")
print(poly_model.coef_)

# Check R-squared
print("Model Accuracy:")
print(poly_model.score(X = predictors, 
                 y = mtcars["mpg"]))