% TODO: rewrite parts of this framework for statically sized datasets

dbgmsg('Starting');

global DEBUG
DEBUG = true;
rng(1);
warning('off','all');

use_parallel = false
try
    %matlabpool open;
    %use_parallel = true
catch
    %disp('Unable to open matlab pool.');
    %use_parallel = false
end
options = statset('UseParallel',use_parallel);

ntimes = 1
nfold = 5

% load linear dataset, X_lin,y_lin
% NOTE: could just make the size, zscore, and mse parts of the load fn
% and return them
[X_lin,y_lin,B_lin,errors_lin]=load_linear_dataset(100,10,0.1,1);
[n_lin, d_lin] = size(X_lin);
ybest = X_lin*B_lin';
ssres = sum((ybest-y_lin).^2);
sstot = sum((y_lin-mean(y_lin)).^2);
best_r2 = 1-(ssres/sstot)
X_lin = zscore(X_lin); % mean center and unit variance
y_lin = zscore(y_lin); % mean center and unit variance
mse0_lin = (1/size(y_lin,1))*sum((y_lin-mean(y_lin)).^2);

% NOTE: doesn't need to be a func for statically sized datasets
cvpart = @(n) cvpartition(n,'kfold',nfold); % where n is the number of rows

% %% Linear
% data_linear = runcomparison( X_lin,y_lin,options,ntimes,'linear',cvpart, d_lin);
% save('linear_results');

% Masked linear
[X_lin,y_lin,B_lin,errors_lin]=load_linear_dataset_masked(400,100,0.3,5,0.15);
[n_lin, d_lin] = size(X_lin);
% This is with the true values
ybest = X_lin*B_lin';
ssres = sum((ybest-y_lin).^2);
sstot = sum((y_lin-mean(y_lin)).^2);
best_r2 = 1-(ssres/sstot) % this is the true

%X_lin = zscore(X_lin); % mean center and unit variance
%y_lin = zscore(y_lin); % mean center and unit variance
mse0_lin = (1/size(y_lin,1))*sum((y_lin-mean(y_lin)).^2);

% Run Lasso
cp = cvpart(size(X_lin,1));
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


% Performing feature selection

Bfitcopy_abs = abs(Bfitorig);
Bfitcopy = Bfitorig;
previous = Bfitorig;
% Try to remove
[mv,mix] = min(abs(Bfitorig));
X_lin_removed = X_lin;
X_lin_removed(:,mix) = 0;

phi = FastfoodForKernel(X_lin_removed',ffen_results.best_para,ffen_results.best_sig,use_spiral)'; % calculate the projections of the training samples
yfit = phi*ffen_results.Bfit;% Bfit; % perform regression

ssres = sum((yfit-y_lin).^2);
sstot = sum((y_lin-mean(y_lin)).^2);
best_R2 = 1-(ssres/sstot);
%prev_MAE = ffen_metrics.MAE;
for i = [0:0.005:0.4]
    Bfitcopy(Bfitcopy_abs < i & Bfitcopy_abs > i-0.005) = 0;

    yfit = X_lin * Bfitcopy;
    ssres = sum((yfit-y_lin).^2);
    sstot = sum((y_lin-mean(y_lin)).^2);
    r2 = 1-(ssres/sstot)
    
    if r2 >= best_R2
       previous = Bfitcopy;
       best_R2 = r2;
    end
    
    %ffen_metrics_removed = evaluate_beta(B_lin',Bfitcopy);
    
   % if prev_MAE >= ffen_metrics_removed.MAE
    %    prev_MAE = ffen_metrics_removed.MAE;
     %   previous = Bfitcopy;

    %end
    
    
        
    Bfitcopy = previous;
end
Bfitorig = Bfitcopy;
   
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


% Below is unknown <----beginning part looks a lot like run_lasso, then it
% runs ffen on cifar-10
% X = X_lin;
% y = y_lin;
% %% Built-in lasso
% accslasso = []; % accuracies
% timslasso = []; % times
% trIdx = ones(1,size(X,1));
% best_alpha = .1;
% best_lambda = .1;
% best_r2 = -Inf;
% % Now we need to divide up into training and testing again,
% % but we will do a very simple percentage to keep things
% % fast
% fracTest = 0.3;
% numTest = round(length(trIdx))*fracT`
% inxs = randperm(length(trIdx));
% testInxs = inxs(1:numTest);
% trainInxs = inxs((numTest+1):end);
% intTrIdx = trIdx(trainInxs);
% intTeIdx = trIdx(testInxs);
% for alpha= .1:.01:.8
%     for lambda= .1:.01:.8
%         ytest = cv_lasso(X(intTrIdx,:),y(intTrIdx),X(intTeIdx,:),alpha,lambda,options);
%         ssres = sum((ytest-y(intTeIdx)).^2);
%         sstot = sum((y(intTeIdx)-mean(y(intTrIdx))).^2);
%         r2 = 1-(ssres/sstot);
%         if (r2 > best_r2)
%             best_alpha = alpha;
%             best_lambda = lambda;
%             best_r2 = r2;
%         end
%     end
% end
% B = lasso(X,y,'alpha',best_alpha,'lambda',best_lambda,'options',options); % perform LASSO on the projections to learn regression coefficients
% yfit = X*B;
% 
% mse = 1/length(ytest)*sum((ytest-y(teIdx)).^2);
% ssres = sum((ytest-y(teIdx)).^2);
% sstot = sum((y(teIdx)-mean(y(trIdx))).^2);
% r2 = 1-(ssres/sstot);
% nmse = mse/mse0;
% 
% 
% 
% % NOTE: doesn't need to be a func for statically sized datasets
% cvpart = @(n) cvpartition(n,'kfold',nfold); % where n is the number of rows
% 
% %% Linear but now masked
% data_linear = runcomparison( X_lin,y_lin,options,ntimes,'linear',cvpart, d_lin);
% save('linear_masked_results');
% 
% % load nonlinear dataset, X_nonlin,y_nonlin
% [X_nonlin,y_nonlin] = load_cifar_10_dataset('/home/kopels/data/cifar-10-batches-mat');
% [n_nonlin, d_nonlin] = size(X_nonlin);
% X_nonlin = zscore(X_nonlin); % mean center and unit variance
% y_nonlin = zscore(y_nonlin); % mean center and unit variance
% mse0_nonlin = (1/size(y_nonlin,1))*sum((y_nonlin-mean(y_nonlin)).^2);
% 
% 
% %% Nonlinear
% data_nonlinear = runcomparison( X_nonlin,y_nonlin,options,ntimes,'nonlinear' );
% 
% %% store results
% dbgmsg('Storing results.');
% lin_lines = {};
% nonlin_lines = {};
% for k = 1:length(n_values)
%     n = n_values(k);
%     for z = 1:length(d_values)
%         d = d_values(z);
%         % linear
%         acclasso = data_linear{k,z}.acclasso;
%         accsven = data_linear{k,z}.accsven;
%         accffen = data_linear{k,z}.accffen;
%         timlasso = data_linear{k,z}.timlasso;
%         timsven = data_linear{k,z}.timsven;
%         timffen = data_linear{k,z}.timffen;
%         % lasso
%         acclasso_nmse_mean = mean(acclasso(:,1));
%         acclasso_nmse_std = std(acclasso(:,1));
%         acclasso_r2_mean = mean(acclasso(:,2));
%         acclasso_r2_std = std(acclasso(:,2));
%         timlasso_mean = mean(timlasso);
%         timlasso_std = std(timlasso);
%         % sven
%         accsven_nmse_mean = mean(accsven(:,1));
%         accsven_nmse_std = std(accsven(:,1));
%         accsven_r2_mean = mean(accsven(:,2));
%         accsven_r2_std = std(accsven(:,2));
%         timsven_mean = mean(timsven);
%         timsvem_std = std(timsven);
%         % ffen
%         accffen_nmse_mean = mean(accffen(:,1));
%         accffen_nmse_std = std(accffen(:,1));
%         accffen_r2_mean = mean(accffen(:,2));
%         accffen_r2_std = std(accffen(:,2));
%         timffen_mean = mean(timffen);
%         timffen_std = std(timffen);
%         line = sprintf('%d,%d,%d,%f,%f,%f,%f,%f,%f\n',...
%             1,n,d,acclasso_r2_mean,acclasso_r2_std,acclasso_nmse_mean,acclasso_nmse_std,...
%             timlasso_mean,timlasso_std);
%         lin_lines{end+1} = line;
%         
%         %nonlinear
%         acclasso = data_nonlinear{k,z}.acclasso;
%         accsven = data_nonlinear{k,z}.accsven;
%         accffen = data_nonlinear{k,z}.accffen;
%         timlasso = data_nonlinear{k,z}.timlasso;
%         timsven = data_nonlinear{k,z}.timsven;
%         timffen = data_nonlinear{k,z}.timffen;
%         % lasso
%         acclasso_nmse_mean = mean(acclasso(:,1));
%         acclasso_nmse_std = std(acclasso(:,1));
%         acclasso_r2_mean = mean(acclasso(:,2));
%         acclasso_r2_std = std(acclasso(:,2));
%         timlasso_mean = mean(timlasso);
%         timlasso_std = std(timlasso);
%         % sven
%         accsven_nmse_mean = mean(accsven(:,1));
%         accsven_nmse_std = std(accsven(:,1));
%         accsven_r2_mean = mean(accsven(:,2));
%         accsven_r2_std = std(accsven(:,2));
%         timsven_mean = mean(timsven);
%         timsvem_std = std(timsven);
%         % ffen
%         accffen_nmse_mean = mean(accffen(:,1));
%         accffen_nmse_std = std(accffen(:,1));
%         accffen_r2_mean = mean(accffen(:,2));
%         accffen_r2_std = std(accffen(:,2));
%         timffen_mean = mean(timffen);
%         timffen_std = std(timffen);
%         line = sprintf('%d,%d,%d,%f,%f,%f,%f,%f,%f\n',...
%             0,n,d,acclasso_r2_mean,acclasso_r2_std,acclasso_nmse_mean,acclasso_nmse_std,...
%             timlasso_mean,timlasso_std);
%         nonlin_lines{end+1} = line;
%     end
% end
% f = fopen('results.csv','wt');
% fprintf(f,'linear,n,d,r2,r2std,nmse,nmsestd,time,timestd\n');
% for i=1:length(lin_lines)
%     fwrite(f,lin_lines{i});
% end
% for i=1:length(nonlin_lines)
%     fwrite(f,nonlin_lines{i});
% end
% dbgmsg('Wrote results to "results.csv". All done.');
% 
% if use_parallel
%     matlabpool close;
% end
% 
% dbgmsg('All done.');