function yfit = cv_lasso( xtrain,ytrain,xtest, alpha,lambda2, options )
% xtrain: rows are samples, columns are variables
% ytrain: vector of labels
% xtest: same as xtrain
% alpha: see docs for LASSO
% lambda2: see docs for LASSO
% options: an instance of statset
% fits a LASSO model and predicts the labels for xtest

B = lasso(xtrain,ytrain,'alpha',alpha,'lambda',lambda2,'options',options); % perform LASSO on the projections to learn regression coefficients
% if options.UseParallel
%     xtest = gpuArray(xtest);
%     B = gpuArray(B);
% end
yfit = xtest*B; % perform regression