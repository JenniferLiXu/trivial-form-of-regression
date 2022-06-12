# trivial-form-of-regression

Your goal is to predict a value y based on a vector x. While the exact relationship is usually not known, in this task, y is the mean of x. You may verify this on the provided training set. Your task is to make predictions for y on the provided test set.

The evaluation metric for this task is the Root Mean Squared Error which is the square root of the mean/average of the square of all of the error.

from sklearn.metrics import mean_squared_error
RMSE = mean_squared_error(y, y_pred)**0.5
