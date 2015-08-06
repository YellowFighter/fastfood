function [yfit,yfittrain] = cv_ffen( xtrain,ytrain,xtest, alpha,lambda2, para,sigma, use_spiral )
% xtrain: rows are samples, columns are variables
% ytrain: vector of labels
% xtest: same as xtrain
% alpha: see docs for LASSO
% lambda2: see docs for LASSO
% fits a FFEN model and predicts the labels for xtest

phitrain = FastfoodForKernel(xtrain',para,sigma,use_spiral)'; % calculate the projections of the training samples
%phitest = FastfoodForKernel(xtest',para,sigma,use_spiral)'; % calculate the projections of the testing samples
%phitrain = phitotal(1:size(xtrain,1),:);
%phitest = phitotal((size(xtrain,1)+1):end,:);
[Bfull,fitinfo] = lasso(phitrain,ytrain,'CV',2);%,'alpha',alpha,'lambda',lambda2); % perform LASSO on the projections to learn regression coefficients
%minMSE = min(fitinfo.MSE);
%ix = find(minMSE == fitinfo.MSE);
ix = find(fitinfo.LambdaMinMSE == fitinfo.Lambda);
B = Bfull(:,ix);
phitest = FastfoodForKernel(xtest',para,sigma,use_spiral)'; % calculate the projections of the testing samples
yfit = phitest*B; % perform regression
yfittrain = phitrain*B; % perform regression
