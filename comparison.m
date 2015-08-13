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

%ntimes = 1
nfold = 5
frac_nonzero = 0.1

n_values = [100,1000,10000]%,100000]
d_values = [100,1000,10000,100000]%,1000000]

y_func_linear = @(X,r) X*r + randn(size(X,1),1)*.1; % linear function
y_func_nonlinear = @(X,r,shuffle) (X(:,shuffle).*X)*r + randn(size(X,1),1)*.1;
cvpart = @(n) cvpartition(n,'kfold',nfold);

%% Linear
disp('Running linear comparison.');
data_linear = {};
for k = 1:length(n_values)
    n = n_values(k);
    for z = 1:length(d_values)
        d = d_values(z);
        fprintf('linear, n = %d, d = %d\n',n,d);
        
        X = randn(n,d);
        r = zeros(size(X,2),1);
        inxs = randperm(d);
        num_nonzero = round(frac_nonzero*d);
        ugly = [-1,1];
        s = [];
        for j = 1:num_nonzero
            ugly_inxs = randperm(2);
            s(j) = ugly(ugly_inxs(1));
        end
        r(inxs(1:num_nonzero)) = s.*(3 + 2*rand(1,num_nonzero));
        y = y_func_linear(X,r);
        X = zscore(X); % mean center and unit variance
        y = zscore(y); % mean center and unit variance
        mse0 = (1/size(y,1))*sum((y-mean(y)).^2);

        %% (Hyper-)params
        lambda2 = 0.1;
        alpha = 0.5;
        t = lambda2*alpha;
        N = d*20; % number of basis functions to use for approximation
        para = FastfoodPara(N,d); % generate FF parameters
        sigma = 10; % band-width of Gaussian kernel
        cp = cvpart(n);

        %% Built-in lasso
        accslasso = []; % accuracies
        timslasso = []; % times
        for l=1:cp.NumTestSets
            fprintf('Linear LASSO n=%d,d=%d l=%d\n',n,d,l);
            trIdx = cp.training(l);
            teIdx = cp.test(l);
            tic;
            ytest = cv_lasso(X(trIdx,:),y(trIdx),X(teIdx,:),alpha,lambda2,options);
            timlasso = toc;
            mse = 1/length(ytest)*sum((ytest-y(teIdx)).^2);
            ssres = sum((ytest-y(teIdx)).^2);
            sstot = sum((y(teIdx)-mean(y(trIdx))).^2);
            r2 = 1-(ssres/sstot);
            nmse = mse/mse0;
            acclasso = [nmse,r2];
            accslasso(l,:) = acclasso;
            timslasso(l) = timlasso;
        end

        %% SVEN
        accssven = []; % accuracies
        timssven = []; % times
        for l=1:cp.NumTestSets
            fprintf('Linear SVEN n=%d,d=%d l=%d\n',n,d,l);
            trIdx = cp.training(l);
            teIdx = cp.test(l);
            tic;
            ytest = cv_sven(X(trIdx,:),y(trIdx),X(teIdx,:),t,lambda2,options);
            timsven = toc;
            mse = 1/length(ytest)*sum((ytest-y(teIdx)).^2);
            ssres = sum((ytest-y(teIdx)).^2);
            sstot = sum((y(teIdx)-mean(y(trIdx))).^2);
            r2 = 1-(ssres/sstot);
            nmse = mse/mse0;
            accsven = [nmse,r2];
            accssven(l,:) = accsven;
            timssven(l) = timsven;
        end


        %% FFEN
        accsffen = []; % accuracies
        timsffen = []; % times
        for l=1:cp.NumTestSets
            fprintf('Linear FFEN n=%d,d=%d l=%d\n',n,d,l);
            trIdx = cp.training(l);
            teIdx = cp.test(l);
            tic;
            ytest = cv_ffen(X(trIdx,:),y(trIdx),X(teIdx,:),alpha,lambda2,para,sigma,use_spiral,options);
            timffen = toc;
            mse = 1/length(ytest)*sum((ytest-y(teIdx)).^2);
            ssres = sum((ytest-y(teIdx)).^2);
            sstot = sum((y(teIdx)-mean(y(trIdx))).^2);
            r2 = 1-(ssres/sstot);
            nmse = mse/mse0;
            accffen = [nmse,r2];
            accsffen(l,:) = accffen;
            timsffen(l) = timffen;
        end

        
        data_linear{k,z} = {};
        data_linear{k,z}.acclasso = accslasso;
        data_linear{k,z}.accsven = accssven;
        data_linear{k,z}.accffen = accsffen;
        data_linear{k,z}.timlasso = timslasso;
        data_linear{k,z}.timsven = timssven;
        data_linear{k,z}.timffen = timsffen;
    end
end

%% Non Linear
disp('Note: the shuffling that occurs to create the nonlinear dataset differs for every (n,d) pair.');
data_nonlinear = {};
for k = 1:length(n_values)
    n = n_values(k);
    for z = 1:length(d_values)
        d = d_values(z);
        fprintf('nonlinear, n = %d, d = %d\n',n,d);
        
        X = randn(n,d);
        shuffle = randperm(d);
        r = zeros(size(X,2),1);
        inxs = randperm(d);
        num_nonzero = round(frac_nonzero*d);
        ugly = [-1,1];
        s = [];
        for j = 1:num_nonzero
            ugly_inxs = randperm(2);
            s(j) = ugly(ugly_inxs(1));
        end
        r(inxs(1:num_nonzero)) = s.*(3 + 2*rand(1,num_nonzero));
        y = y_func_nonlinear(X,r,shuffle);
        X = zscore(X); % mean center and unit variance
        y = zscore(y); % mean center and unit variance
        mse0 = (1/size(y,1))*sum((y-mean(y)).^2);

        %% (Hyper-)params
        lambda2 = 0.1;
        alpha = 0.5;
        t = lambda2*alpha;
        N = d*20; % number of basis functions to use for approximation
        para = FastfoodPara(N,d); % generate FF parameters
        sigma = 10; % band-width of Gaussian kernel
        cp = cvpart(n);

        %% Built-in lasso
        accslasso = []; % accuracies
        timslasso = []; % times
        for l=1:cp.NumTestSets
            fprintf('NonLinear LASSO n=%d,d=%d l=%d\n',n,d,l);
            trIdx = cp.training(l);
            teIdx = cp.test(l);
            tic;
            ytest = cv_lasso(X(trIdx,:),y(trIdx),X(teIdx,:),alpha,lambda2,options);
            timlasso = toc;
            mse = 1/length(ytest)*sum((ytest-y(teIdx)).^2);
            ssres = sum((ytest-y(teIdx)).^2);
            sstot = sum((y(teIdx)-mean(y(trIdx))).^2);
            r2 = 1-(ssres/sstot);
            nmse = mse/mse0;
            acclasso = [nmse,r2];
            accslasso(l,:) = acclasso;
            timslasso(l) = timlasso;
        end

        %% SVEN
        accssven = []; % accuracies
        timssven = []; % times
        for l=1:cp.NumTestSets
            fprintf('NonLinear SVEN n=%d,d=%d l=%d\n',n,d,l);
            trIdx = cp.training(l);
            teIdx = cp.test(l);
            tic;
            ytest = cv_sven(X(trIdx,:),y(trIdx),X(teIdx,:),t,lambda2,options);
            timsven = toc;
            mse = 1/length(ytest)*sum((ytest-y(teIdx)).^2);
            ssres = sum((ytest-y(teIdx)).^2);
            sstot = sum((y(teIdx)-mean(y(trIdx))).^2);
            r2 = 1-(ssres/sstot);
            nmse = mse/mse0;
            accsven = [nmse,r2];
            accssven(l,:) = accsven;
            timssven(l) = timsven;
        end


        %% FFEN
        accsffen = []; % accuracies
        timsffen = []; % times
        for l=1:cp.NumTestSets
            fprintf('NonLinear FFEN n=%d,d=%d l=%d\n',n,d,l);
            trIdx = cp.training(l);
            teIdx = cp.test(l);
            tic;
            ytest = cv_ffen(X(trIdx,:),y(trIdx),X(teIdx,:),alpha,lambda2,para,sigma,use_spiral,options);
            timffen = toc;
            mse = 1/length(ytest)*sum((ytest-y(teIdx)).^2);
            ssres = sum((ytest-y(teIdx)).^2);
            sstot = sum((y(teIdx)-mean(y(trIdx))).^2);
            r2 = 1-(ssres/sstot);
            nmse = mse/mse0;
            accffen = [nmse,r2];
            accsffen(l,:) = accffen;
            timsffen(l) = timffen;
        end

        
        data_nonlinear{k,z} = {};
        data_nonlinear{k,z}.acclasso = accslasso;
        data_nonlinear{k,z}.accsven = accssven;
        data_nonlinear{k,z}.accffen = accsffen;
        data_nonlinear{k,z}.timlasso = timslasso;
        data_nonlinear{k,z}.timsven = timssven;
        data_nonlinear{k,z}.timffen = timsffen;
    end
end

%% store results
disp('Storing results.');
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
disp('Wrote results to "results.csv". All done.');

if use_parallel
    matlabpool close;
end