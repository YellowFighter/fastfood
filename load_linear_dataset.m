function [X,y,B,errors] = load_linear_dataset(n,d,noise_stdev,xmax)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

signs = randi(2,1,d);
signs(find(signs==2)) = -1;
B = signs.*rand(1,d);

errors = noise_stdev*randn(n,1);

X = xmax*rand(n,d);

y = X*B' + errors;

end

