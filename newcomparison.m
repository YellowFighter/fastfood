rng(3,'twister'); % for reproducibility
niter = 10; % number of CV iterations
% Load training data.
load('/Volumes/Development/datasets/MNIST/mnist_all.mat');
test = double([test0;test1;test2;test3;test4;test5;test6;test7;test8;test9]);
test_labels = [repmat(0,size(test0,1),1);repmat(1,size(test1,1),1);repmat(2,size(test2,1),1);repmat(3,size(test3,1),1);repmat(4,size(test4,1),1);repmat(5,size(test5,1),1);repmat(6,size(test6,1),1);repmat(7,size(test7,1),1);repmat(8,size(test8,1),1);repmat(9,size(test9,1),1)];
train = double([train0;train1;train2;train3;train4;train5;train6;train7;train8;train9]);
train_labels = [repmat(0,size(train0,1),1);repmat(1,size(train1,1),1);repmat(2,size(train2,1),1);repmat(3,size(train3,1),1);repmat(4,size(train4,1),1);repmat(5,size(train5,1),1);repmat(6,size(train6,1),1);repmat(7,size(train7,1),1);repmat(8,size(train8,1),1);repmat(9,size(train9,1),1)];
% Fit a CV sequence of models with lasso and plot result
[b fitinfo] = lasso(train,train_labels,'CV',niter,'Options',statset('Display','iter'));
lassoPlot(b,fitinfo,'PlotType','Lambda','XScale','log');