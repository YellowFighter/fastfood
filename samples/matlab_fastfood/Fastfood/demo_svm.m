% Load data.
X = csvread('~/dev/fastfood/data_sets/kddcup99/X.csv');
[N,D] = size(X); % size of input data
fprintf('Loaded %d samples of dimensionality %d\n',N,D);

% Train SVM model with Fastfood RBF kernel.
n = D*20; % number of basis functions in approximation
fprintf('Using %d basis functions.\n',n);
sgm = 10; % bandwidth of gaussian kernel
para = FastfoodPara(n,D);
%phi = FastfoodForKernel(Xtrain', para, sgm, false)';
phi = FastfoodForKernel(X', para, sgm, false)';
csvwrite('~/dev/fastfood/data_sets/kddcup99/phi.csv',phi);
fprintf('Dimensions of transformed data: %dx%d\n',size(phi));