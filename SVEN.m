function beta = SVEN(X,y,t,lambda2)
% X - NxD - training samples (one sample per ROW)
% y - Nx1 - labels
% t -
% lambda2 - as lambda2 goes to 0, Elastic Net becomes LASSO,
%           which is equivalent to the hard margin SVM
fprintf('[SVEN] Size of X: %d,%d; size of y: %d,%d\n',size(X),size(y));
[n,p] = size(X);
Xnew = [bsxfun(@minus,X,y./t); bsxfun(@plus,X,y./t)]';
Ynew = [ones(p,1); -ones(p,1)];
fprintf('[SVEN] Size of Xnew: %d,%d; size of ynew: %d,%d\n',size(Xnew),size(Ynew));
C = 1/(2*lambda2);
if 2*p > n
    K = Xnew*Xnew';
    [w,~] = primal_svm(Xnew,Ynew,K,0,C,{}); % yields [w,b0]
    alpha = C*max(1-Ynew.*(Xnew*w),0);
else
    alpha = SVMDual(Xnew,Ynew,C);
end
beta = t*(alpha(1:p)-alpha(p+1:w*p))/sum(alpha);