%Getting started
dbgmsg('Starting');
global DEBUG
DEBUG = true;
rng(1);
warning('off','all');
use_parallel = false;
options = statset('UseParallel',use_parallel);

%load the dataset
[X_wine, y_wine] = load_wine_dataset('/home/kopels/data/wine-white-no-labels.csv');
y_wine_scaled = zscore(y_wine);
X_wine_scaled = zscore(X_wine);

%Lasso
%cp = cvpart(size(X_wine,1));
lambdas = 0.1:0.05:0.9;
alphas = 0.1:0.05:0.9;
lasso_results = run_lasso(X_wine_scaled,y_wine_scaled,{},lambdas,alphas,options);
yfit_unscaled = round((lasso_results.yfit + 1)/2*(max(y_wine)-min(y_wine))+min(y_wine));
acc = length(find(yfit_unscaled == y_wine))/length(y_wine);
%No true model to compare to. Set up CV?
%lasso_metrics_wine = evaluate_beta(B_lin',lasso_results.Bfit)


%FFEN
sigmas = [1,5,10];
use_spiral = 0;
Ns = [1,10,20]*size(X_wine,2);
ffen_results = run_ffen(X_wine_scaled,y_wine_scaled,{},.1,.1,sigmas,Ns,use_spiral,options);
yfit_unscaled = round((ffen_results.yfit + 1)/2*(max(y_wine)-min(y_wine))+min(y_wine));
acc = length(find(yfit_unscaled == y_wine))/length(y_wine);

%Reduce to original space
rho = corr(X_wine,ffen_results.phi);
Bfitorig = rho*ffen_results.Bfit;
%Bfitorig = Bfitorig/norm(Bfitorig)*norm(B_lin);
phi = FastfoodForKernel(X_wine',ffen_results.best_para,ffen_results.best_sig,use_spiral)'; % calculate the projections of the training samples
ytest_ffen= phi*ffen_results.Bfit;

%ffen_metrics = evaluate_beta(B_lin',Bfitorig);


%Calculating MSE
MSE_ffen = sum((ytest_ffen-y_wine).^2)
MSE_lasso = sum((lasso_results.yfit-y_wine).^2)
avg_MSE_ffen = MSE_ffen/size(ytest_ffen,1)
avg_MSE_lasso = MSE_lasso/size(ytest_ffen,1)