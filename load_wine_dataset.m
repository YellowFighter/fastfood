function [X,y] = load_wine_dataset(path)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
matrix = csvread(path);
y = matrix(:,12);
X = matrix(:,1:11);

end

