warning('off','all'); % turn off all warnings
rng(0); % seed random number generator
try
    matlabpool open;
    use_parallel = true;
catch
    display('Unable to open matlab pool')
    use_parallel = false;
end

%% Generate/load data
% generate a linear dataset
n = 1000;
d = 150;
X = randn(n,d);
r = zeros(size(X,2),1);
r(1:5)= [0;3;0;-3;0]; % only two nonzero coefficients (2,4)
y = X*r + randn(size(X,1),1)*.1; % small added noise
X = zscore(X); % mean centering and unit variance

%% (Hyper-)params
lambda2 = 0.1; % see docs for LASSO for lambda2, alpha
alpha = 0.5;
t = lambda2*alpha; % SVEN parameter
cp = cvpartition(n,'kfold',5); % create the 5-fold partitions
N = d*20; % number of basis functions to use for FF approximation
sigma = 10; % band-width of Gaussian kernel
options = statset('UseParallel',use_parallel);

%% Built-in lasso
B = lasso(X,y,'alpha',alpha,'lambda',lambda2);
find(B(:,1) ~= 0) % should only return 2,4
mse = crossval('mse',X,y,'partition',cp,...
    'Predfun',@(xtrain,ytrain,xtest) cv_lasso(xtrain,ytrain,xtest,alpha,lambda2)); % perform CV to get a MSE
printf('MSE for built-in LASSO: %f\n',mse);

%% SVEN
beta = SVEN(X',y',t,lambda2);
find(beta ~= 0) % should only return 2,4
mse = crossval('mse',X,y,'partition',cp,...
    'Predfun',@(xtrain,ytrain,xtest) cv_sven(xtrain,ytrain,xtest,t,lambda2)); % perform CV to get a MSE
printf('MSE for SVEN: %f\n',mse);

%% FFEN
mse = crossval('mse',X,y,'partition',cp,...
    'Predfun',@(xtrain,ytrain,xtest) cv_ffen(xtrain,ytrain,xtest,alpha,lambda2)); % perform CV to get a MSE
printf('MSE for FFEN: %f\n',mse);