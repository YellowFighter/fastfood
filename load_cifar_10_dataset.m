function [X,y] = load_cifar_10_dataset(path)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

matrix = load([path,'/','data_batch_1']);
y = double(matrix.labels);
X = double(matrix.data);

for i = 2:5
    matrix = load([path,'/','data_batch_',num2str(i)]);
    y = [y;double(matrix.labels)]; 
    X = [X;double(matrix.data)]; 
end

end
