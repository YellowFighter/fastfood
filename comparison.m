dbgmsg('Starting');

global DEBUG
DEBUG = true;
rng(1);
warning('off','all');
try % test whether we can use Spiral package
    fwht_spiral([1; 1]);
    use_spiral = 1
catch
    use_spiral = 0
end
try
    matlabpool open;
    use_parallel = true
catch
     disp('Unable to open matlab pool.');
     use_parallel = false
end
options = statset('UseParallel',use_parallel);

ntimes = 1
nfold = 5

% load linear dataset, X_lin,y_lin
[X_lin,y_lin] = load_linear_dataset()
n_lin, d_lin = size(X_lin)
X_lin = zscore(X_lin); % mean center and unit variance
y_lin = zscore(y_lin); % mean center and unit variance
mse0_lin = (1/size(y_lin,1))*sum((y_lin-mean(y_lin)).^2);

% load nonlinear dataset, X_lin,y_lin
[X_nonlin,y_nonlin] = load_nonlinear_dataset()
n_nonlin, d_nonlin = size(X_nonlin)
X_nonlin = zscore(X_nonlin); % mean center and unit variance
y_nonlin = zscore(y_nonlin); % mean center and unit variance
mse0_nonlin = (1/size(y_nonlin,1))*sum((y_nonlin-mean(y_nonlin)).^2);

cvpart = @(n) cvpartition(n,'kfold',nfold);

%% Linear
data_linear = runcomparison( X_lin,y_nonlin,options,ntimes,'linear' );

%% Nonlinear
data_nonlinear = runcomparison( X_nonlin,y_nonlin,options,ntimes,'nonlinear' );

%% store results
dbgmsg('Storing results.');
lin_lines = {};
nonlin_lines = {};
for k = 1:length(n_values)
    n = n_values(k);
    for z = 1:length(d_values)
        d = d_values(z);
        % linear
        acclasso = data_linear{k,z}.acclasso;
        accsven = data_linear{k,z}.accsven;
        accffen = data_linear{k,z}.accffen;
        timlasso = data_linear{k,z}.timlasso;
        timsven = data_linear{k,z}.timsven;
        timffen = data_linear{k,z}.timffen;
        % lasso
        acclasso_nmse_mean = mean(acclasso(:,1));
        acclasso_nmse_std = std(acclasso(:,1));
        acclasso_r2_mean = mean(acclasso(:,2));
        acclasso_r2_std = std(acclasso(:,2));
        timlasso_mean = mean(timlasso);
        timlasso_std = std(timlasso);
        % sven
        accsven_nmse_mean = mean(accsven(:,1));
        accsven_nmse_std = std(accsven(:,1));
        accsven_r2_mean = mean(accsven(:,2));
        accsven_r2_std = std(accsven(:,2));
        timsven_mean = mean(timsven);
        timsvem_std = std(timsven);
        % ffen
        accffen_nmse_mean = mean(accffen(:,1));
        accffen_nmse_std = std(accffen(:,1));
        accffen_r2_mean = mean(accffen(:,2));
        accffen_r2_std = std(accffen(:,2));
        timffen_mean = mean(timffen);
        timffen_std = std(timffen);
        line = sprintf('%d,%d,%d,%f,%f,%f,%f,%f,%f\n',...
            1,n,d,acclasso_r2_mean,acclasso_r2_std,acclasso_nmse_mean,acclasso_nmse_std,...
            timlasso_mean,timlasso_std);
        lin_lines{end+1} = line;
        
        %nonlinear
        acclasso = data_nonlinear{k,z}.acclasso;
        accsven = data_nonlinear{k,z}.accsven;
        accffen = data_nonlinear{k,z}.accffen;
        timlasso = data_nonlinear{k,z}.timlasso;
        timsven = data_nonlinear{k,z}.timsven;
        timffen = data_nonlinear{k,z}.timffen;
        % lasso
        acclasso_nmse_mean = mean(acclasso(:,1));
        acclasso_nmse_std = std(acclasso(:,1));
        acclasso_r2_mean = mean(acclasso(:,2));
        acclasso_r2_std = std(acclasso(:,2));
        timlasso_mean = mean(timlasso);
        timlasso_std = std(timlasso);
        % sven
        accsven_nmse_mean = mean(accsven(:,1));
        accsven_nmse_std = std(accsven(:,1));
        accsven_r2_mean = mean(accsven(:,2));
        accsven_r2_std = std(accsven(:,2));
        timsven_mean = mean(timsven);
        timsvem_std = std(timsven);
        % ffen
        accffen_nmse_mean = mean(accffen(:,1));
        accffen_nmse_std = std(accffen(:,1));
        accffen_r2_mean = mean(accffen(:,2));
        accffen_r2_std = std(accffen(:,2));
        timffen_mean = mean(timffen);
        timffen_std = std(timffen);
        line = sprintf('%d,%d,%d,%f,%f,%f,%f,%f,%f\n',...
            0,n,d,acclasso_r2_mean,acclasso_r2_std,acclasso_nmse_mean,acclasso_nmse_std,...
            timlasso_mean,timlasso_std);
        nonlin_lines{end+1} = line;
    end
end
f = fopen('results.csv','wt');
fprintf(f,'linear,n,d,r2,r2std,nmse,nmsestd,time,timestd\n');
for i=1:length(lin_lines)
    fwrite(f,lin_lines{i});
end
for i=1:length(nonlin_lines)
    fwrite(f,nonlin_lines{i});
end
dbgmsg('Wrote results to "results.csv". All done.');

if use_parallel
    matlabpool close;
end

dbgmsg('All done.');