% Load data.
%X = csvread('~/dev/fastfood/digits-2.csv');
X = csvread('~/dev/fastfood/x_f.csv');
Y = csvread('~/dev/fastfood/digits-2-y.csv');
Y = logical(Y);
[N,D] = size(X); % size of input data
fprintf('Loaded %d samples of dimensionality %d\n',N,D);

% Center and scale data.
%X = X-mean(X(:)); % mean 0
%X = X/std(X(:));  % stdev 1

% Shuffle rows.

% Split train and test data.
% Using 80-20
trainidxs = randperm(0.8*N);
testidxs = setdiff(1:N, trainidxs);
Xtrain = X(trainidxs,:);
Xtest = X(testidxs,:);
Ytrain = Y(trainidxs,:);
Ytest = Y(testidxs,:);

% Train SVM model with standard RBF kernel.
% defaults to rbf sigma = 1
disp('Training standard model');
tic
model = svmtrain(Xtrain,Ytrain,...
    'kernel_function','rbf');
tm_std = toc;
fprintf('Elapsed time to train std model %f seconds\n',tm_std);

% Test performance of standard model.
disp('Classifying with standard model.');
tic
Yhat = svmclassify(model,Xtest);
toc
mcr = sum(Yhat ~= Ytest) / size(Ytest,1);
fprintf('mcr for std model %f\n',mcr);

% Train SVM model with Fastfood RBF kernel.
n = D*20; % number of basis functions in approximation
fprintf('Using %d basis functions.\n',n);
sgm = 10; % bandwidth of gaussian kernel
para = FastfoodPara(n,D);
%phi = FastfoodForKernel(Xtrain', para, sgm, false)';
phi = FastfoodForKernel(X', para, sgm, false)';
csvwrite('phi.csv',phi);
fprintf('Dimensions of transformed data: %dx%d\n',size(phi));
disp('Training Fastfood model.')
tic
ffmodel = svmtrain(phi,Ytrain,...
    'kernel_function','linear');
tm_ff = toc;
fprintf('Elapsed time to train ff model %f seconds\n',tm_std);

% Test performance of fastfood model.
disp('Classifying with fastfood model.');
tic
Yhat = svmclassify(model,Xtest);
toc
mcr = sum(Yhat ~= Ytest) / size(Ytest,1);
fprintf('mcr for ff model %f\n',mcr);

fprintf('FF model took %f (%f%%) seconds LONGER than std svm model\n',...
    tm_ff-tm_std,100*(tm_ff-tm_std)/mean([tm_ff,tm_std]));