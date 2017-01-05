function [X,y] = load_park_dataset(path)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
ds = csvread(path,2,1);
y = ds(:,6);
X = ds(:,[2:4,7:end]);

end

