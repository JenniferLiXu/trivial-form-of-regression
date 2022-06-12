# regression

Task 0: trivial-form-of-regression
Your goal is to predict a value y based on a vector x. While the exact relationship is usually not known, in this task, y is the mean of x. You may verify this on the provided training set. Your task is to make predictions for y on the provided test set.

The evaluation metric for this task is the Root Mean Squared Error which is the square root of the mean/average of the square of all of the error.

from sklearn.metrics import mean_squared_error
RMSE = mean_squared_error(y, y_pred)**0.5


Task 1a: Cross-validation for Ridge Regression
Using cross-validation for ridge regression. perform 10-fold cross-validation with ridge regression for each value of 
λ given above and report the Root Mean Squared Error (RMSE) averaged over the 10 test folds. In other words, for each λ, you should train a ridge regression 10 times leaving out a different fold each time, and report the average of the RMSEs on the left-out folds. 


Task 1b: linear regression
given an input vector x, your goal is to predict a value y as a linear function of a set of feature transformations, ϕ(x).
