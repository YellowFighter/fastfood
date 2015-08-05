function yfit = cv_ffen( xtrain,ytrain,xtest, alpha,lambda2, para,sigma )
% xtrain: rows are samples, columns are variables
% ytrain: vector of labels
% xtest: same as xtrain
% alpha: see docs for LASSO
% lambda2: see docs for LASSO
% fits a FFEN model and predicts the labels for xtest

try % test whether we can use Spiral package
    fwht_spiral([1; 1]);
    use_spiral = 1;
catch
    use_spiral = 0;
end
phitrain = FastfoodForKernel(xtrain',para,sigma,use_spiral)'; % calculate the projections of the training samples
B = lasso(phitrain,ytrain,'alpha',alpha,'lambda',lambda2); % perform LASSO on the projections to learn regression coefficients
phitest = FastfoodForKernel(xtest',para,sigma,use_spiral)'; % calculate the projections of the testing samples
yfit = phitest*B; % perform regression