function yfit = cv_lasso( xtrain,ytrain,xtest, alpha,lambda2 )
% mse = crossval('mse',X,y,'Predfun',predfun) returns mse, a scalar
% containing a 10-fold cross validation estimate of mean-squared error for
% the function predfun. X can be a column vector, matrix, or array of
% predictors. y is a column vector of response values. X and y must have
% the same number of rows.
% 
% predfun is a function handle called with the training subset of X, the
% training subset of y, and the test subset of X as follows:
% 
% yfit = predfun(XTRAIN,ytrain,XTEST)
% Each time it is called, predfun should use XTRAIN and ytrain to fit a
% regression model and then return fitted values in a column vector yfit.
% Each row of yfit contains the predicted values for the corresponding row
% of XTEST. crossval computes the squared errors between yfit and the
% corresponding response test set, and returns the overall mean across
% all test sets.

B = lasso(xtrain,ytrain,'alpha',alpha,'lambda',lambda2);
yfit = xtest*B;