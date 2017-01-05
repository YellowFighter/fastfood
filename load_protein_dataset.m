function [X,y] = load_protein_dataset(path)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
matrix = csvread(path,2);
y = matrix(:,1);
X = matrix(:,2:end);

end