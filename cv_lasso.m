function yfit = cv_lasso( xtrain,ytrain,xtest, alpha,lambda2 )
% xtrain: rows are samples, columns are variables
% ytrain: vector of labels
% xtest: same as xtrain
% alpha: see docs for LASSO
% lambda2: see docs for LASSO
% options: an instance of statset
% fits a LASSO model and predicts the labels for xtest

[Bfull,fitinfo] = lasso(xtrain,ytrain,'CV',2);%,'alpha',alpha,'lambda',lambda2); % perform LASSO on the projections to learn regression coefficients
%minMSE = min(fitinfo.MSE);
%ix = find(minMSE == fitinfo.MSE);
ix = find(fitinfo.LambdaMinMSE == fitinfo.Lambda);
B = Bfull(:,ix);
%B = lasso(xtrain,ytrain,'alpha',alpha,'lambda',lambda2); % perform LASSO on the projections to learn regression coefficients
yfit = xtest*B; % perform regression
