function alpha = SVMDual(X,Y,C)
fprintf('[SVMDual] Size of X: %d,%d; size of y: %d,%d\n',size(X),size(Y));
svm_struct = svmtrain(X,Y,'kernel_function','rbf','rbf_sigma',10,...
    'boxconstraint',C);
alpha = svm_struct.Alpha;