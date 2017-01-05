function [X,y,B,errors] = load_linear_dataset_masked(n,d,noise_stdev,xmax,fracMask)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

signs = randi(2,1,d);
signs(find(signs==2)) = -1;
B = signs.*rand(1,d);

shuffledOrder = randperm(d);
maskInxs = shuffledOrder(1:round(d*fracMask));
B(maskInxs) = 0;

errors = noise_stdev*randn(n,1);

X = xmax*(2*rand(n,d)-1);

y = X*B' + errors;



end

