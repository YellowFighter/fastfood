X = randn(1000,150);
r = zeros(size(X,2),1);
r(1:5)= [0;3;0;-3;0]; % only two nonzero coefficients (2,4)
y = X*r + randn(size(X,1),1)*.1; % small added noise

lambda2 = 0.1;
alpha = 0.5;
t = lambda2*alpha;

B = lasso(X,y,'alpha',alpha,'lambda',lambda2);
find(B(:,1) ~= 0)

beta = SVEN(X',y',t,lambda2);
find(beta ~= 0)
