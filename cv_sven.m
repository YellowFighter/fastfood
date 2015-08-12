function yfit = cv_sven( xtrain,ytrain,xtest, t,lambda2, options )
% xtrain: rows are samples, columns are variables
% ytrain: vector of labels
% xtest: same as xtrain
% t: number of features selected (see SVEN paper)
% lambda2: see docs for LASSO
% fits a SVEN model and predicts the labels for xtest

beta = SVEN(xtrain',ytrain',t,lambda2); % perform SVEN on the projections to learn regression coefficients
if options.UseParallel
    xtest = gpuArray(xtest);
    beta = gpuArray(beta);
end
yfit = xtest*beta; % perform regression