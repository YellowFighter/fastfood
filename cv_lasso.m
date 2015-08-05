function yfit = cv_lasso( xtrain,ytrain,xtest, alpha,lambda2, options )
% xtrain: rows are samples, columns are variables
% ytrain: vector of labels
% xtest: same as xtrain
% alpha: see docs for LASSO
% lambda2: see docs for LASSO
% options: an instance of statset
% fits a LASSO model and predicts the labels for xtest

B = lasso(xtrain,ytrain,'alpha',alpha,'lambda',lambda2,...
    'Options',options); % perform LASSO on the projections to learn regression coefficients
yfit = xtest*B; % perform regression