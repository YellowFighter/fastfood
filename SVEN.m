function beta = SVEN(X,y,t,lambda2)
% X - n,p - where p is the number of samples
% y - p - labels
% t -
% lambda2 - as lambda2 goes to 0, Elastic Net becomes LASSO,
%           which is equivalent to the hard margin SVM
%fprintf('[SVEN] Size of X: %d,%d; size of y: %d,%d\n',size(X),size(y));
[n,p] = size(X); % n is number of variables, p is the number of samples
Xnew = [bsxfun(@minus,X,y./t);bsxfun(@plus,X,y./t)]';
Ynew = [ones(n,1); -ones(n,1)];
%fprintf('[SVEN] Size of Xnew: %d,%d; size of ynew: %d,%d\n',size(Xnew),size(Ynew));
C = 1/(2*lambda2);
global K
global X
K = Xnew'*Xnew;
X = Xnew;
[w,b] = primal_svm(X',Ynew,K,0,C,{}); % yields [w,b0]
alpha = C*max([1-Ynew.*((K*w)+b),zeros(size(Ynew))],[],2);
beta = t*(alpha(1:n)-alpha(n+1:(2*n)))/sum(alpha);