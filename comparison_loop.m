%Getting started
dbgmsg('Starting');
global DEBUG
DEBUG = true;
rng(1);
warning('off','all');
use_parallel = false;
options = statset('UseParallel',use_parallel);

%Run an initial time
%Generate the data
[X_lin,y_lin,B_lin,errors_lin]=load_linear_dataset_masked(5000,10,0.3,1,0.2);
[n_lin, d_lin] = size(X_lin);
% This is with the true values
ybest = X_lin*B_lin';
ssres = sum((ybest-y_lin).^2);
sstot = sum((y_lin-mean(y_lin)).^2);
best_r2 = 1-(ssres/sstot); % this is the true
mse0_lin = (1/size(y_lin,1))*sum((y_lin-mean(y_lin)).^2);

% Run Lasso
lambdas = 0.1:0.05:0.9;
alphas = 0.1:0.05:0.9;
lasso_results = run_lasso(X_lin,y_lin,{},lambdas,alphas,options);

lasso_metrics = evaluate_beta(B_lin',lasso_results.Bfit);

% Run FFEN
sigmas = [1,5,10];
use_spiral = 0;
Ns = [1,10,20]*size(X_lin,2);
ffen_results = run_ffen(X_lin,y_lin,{},lambdas,alphas,sigmas,Ns,use_spiral,options);

% Now convert back to original space
rho = corr(X_lin,ffen_results.phi);
Bfitorig = rho*ffen_results.Bfit;
Bfitorig = Bfitorig/norm(Bfitorig)*norm(B_lin);

ffen_metrics = evaluate_beta(B_lin',Bfitorig);

% Try to remove
[mv,mix] = min(abs(Bfitorig));
X_lin_removed = X_lin;
X_lin_removed(:,mix) = 0;

phi = FastfoodForKernel(X_lin_removed',ffen_results.best_para,ffen_results.best_sig,use_spiral)'; % calculate the projections of the training samples
yfit = phi*ffen_results.Bfit;% Bfit; % perform regression

ssres = sum((yfit-y_lin).^2);
sstot = sum((y_lin-mean(y_lin)).^2);
r2 = 1-(ssres/sstot);

MAEs = [ffen_metrics.MAE, lasso_metrics.MAE];
results = [MAEs]

%loop to test on however many random datasets
results = [];
num_runs = 20;

for i = 1:num_runs
    %Generate the data
    [X_lin,y_lin,B_lin,errors_lin]=load_linear_dataset_masked(5000,10,0.3,1,0.2);
    [n_lin, d_lin] = size(X_lin);
    % This is with the true values
    ybest = X_lin*B_lin';
    ssres = sum((ybest-y_lin).^2);
    sstot = sum((y_lin-mean(y_lin)).^2);
    best_r2 = 1-(ssres/sstot); % this is the true
    mse0_lin = (1/size(y_lin,1))*sum((y_lin-mean(y_lin)).^2);

    % Run Lasso
    lambdas = 0.1:0.05:0.9;
    alphas = 0.1:0.05:0.9;
    lasso_results = run_lasso(X_lin,y_lin,{},lambdas,alphas,options);
    
    lasso_metrics = evaluate_beta(B_lin',lasso_results.Bfit);
    
    % Run FFEN
    sigmas = [1,5,10];
    use_spiral = 0;
    Ns = [1,10,20]*size(X_lin,2);
    ffen_results = run_ffen(X_lin,y_lin,{},lambdas,alphas,sigmas,Ns,use_spiral,options);
    
    % Now convert back to original space
    rho = corr(X_lin,ffen_results.phi);
    Bfitorig = rho*ffen_results.Bfit;
    Bfitorig = Bfitorig/norm(Bfitorig)*norm(B_lin);
    
    ffen_metrics = evaluate_beta(B_lin',Bfitorig);
    
    % Try to remove
    [mv,mix] = min(abs(Bfitorig));
    X_lin_removed = X_lin;
    X_lin_removed(:,mix) = 0;
    
    phi = FastfoodForKernel(X_lin_removed',ffen_results.best_para,ffen_results.best_sig,use_spiral)'; % calculate the projections of the training samples
    yfit = phi*ffen_results.Bfit;% Bfit; % perform regression
    
    ssres = sum((yfit-y_lin).^2);
    sstot = sum((y_lin-mean(y_lin)).^2);
    r2 = 1-(ssres/sstot);
    
    MAEs = [ffen_metrics.MAE, lasso_metrics.MAE];
    results = vertcat(results,MAEs)
    
end

%stored as ffen, lasso
results