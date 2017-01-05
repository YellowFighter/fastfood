function [X,y] = load_kegg_dataset(path,yinx)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
ds = dataset('file',path,'delimiter',',','TreatAsEmpty','?','ReadVarNames',false);
y = double(ds(:,yinx));
X = double(ds(:,[2:(yinx-1),(yinx+1):end]));

end
