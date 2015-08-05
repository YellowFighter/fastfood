rng(1);
warning('off','all');
try
    matlabpool open;
    use_parallel = true;
catch
    display('Can not open matlab pool.');
    use_parallel = false;
end
options = statset('UseParallel',use_parallel);

ntimes = 10;
frac_nonzero = 0.1;

n_values = [100,1000,10000];%,100000];
d_values = [150,300,600,1000,10000];
accuracy_data = {};

y_func = @(X,r) (X*r + randn(size(X,1),1)*.1); % linear function

for k = 1:length(n_values)
    n = n_values(k);
    for z = 1:length(d_values)
        d = d_values(z);
        fprintf('data n = %d, d = %d\n',n,d);
        
        acclasso = [];
        accsven = [];
        accffen = [];
        for i = 1:ntimes
            fprintf('\titer %d\n',i);
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
            y = y_func(X,r);
            X = zscore(X); % mean center and unit variance
            y = zscore(y); % mean center and unit variance
            mse0 = (1/size(y,1))*sum((y-mean(y)).^2);
            
            %% (Hyper-)params
            lambda2 = 0.1;
            alpha = 0.5;
            t = lambda2*alpha;
            N = d*20; % number of basis functions to use for approximation
            para = FastfoodPara(N,d); % generate FF parameters
            cp = cvpartition(n,'kfold',5); % create the 5-fold partitions
            
            %% Built-in lasso
            mses = [];
            r2s = [];
            for j = 1:cp.NumTestSets
                trIdx = cp.training(j);
                teIdx = cp.test(j);
                ytest = cv_lasso(X(trIdx,:),y(trIdx),X(teIdx,:),alpha,lambda2,options);
                % MSE
                mses(j) = 1/length(ytest)*sum((ytest-y(teIdx)).^2);
                
                % R^2
                ssres = sum((ytest-y(teIdx)).^2);
                sstot = sum((y(teIdx)-mean(y(trIdx))).^2);
                r2 = 1-(ssres/sstot);
                r2s(j) = r2;
            end
            mse = mean(mses);
            nmse = mse/mse0;
            r2 = mean(r2s);
            %mse = crossval('mse',X,y,'partition',cp,...
            %    'Predfun',@(xtrain,ytrain,xtest) cv_lasso(xtrain,ytrain,xtest,alpha,lambda2,options)); % perform CV to get a MSE
            acclasso(i,:) = [nmse,r2];
            fprintf('\t\tacclasso nmse = %f, r^2 = %f\n',acclasso(i,1),acclasso(i,2));
            
            %% SVEN
            mses = [];
            r2s = [];
            for j = 1:cp.NumTestSets
                trIdx = cp.training(j);
                teIdx = cp.test(j);
                ytest = cv_sven(X(trIdx,:),y(trIdx),X(teIdx,:),t,lambda2);
                mses(j) = 1/length(ytest)*sum((ytest-y(teIdx)).^2); % MSE
                % R^2
                ssres = sum((y(teIdx)-ytest).^2);
                sstot = sum((y(teIdx)-mean(y(trIdx))).^2);
                r2 = 1-(ssres/sstot);
                r2s(j) = r2;
            end
            mse = mean(mses);
            nmse = mse/mse0;
            r2 = mean(r2s);
%             mse = crossval('mse',X,y,'partition',cp,...
%                 'Predfun',@(xtrain,ytrain,xtest) cv_sven(xtrain,ytrain,xtest,t,lambda2)); % perform CV to get a MSE
            accsven(i,:) = [nmse,r2];
            fprintf('\t\taccsven nmse = %f, r^2 = %f\n',accsven(i,1),accsven(i,2));
            
            %% FFEN
            mses = [];
            r2s = [];
            for j = 1:cp.NumTestSets
                trIdx = cp.training(j);
                teIdx = cp.test(j);
                ytest = cv_ffen(X(trIdx,:),y(trIdx),X(teIdx,:),alpha,lambda2,options);
                % MSE
                mses(j) = 1/length(ytest)*sum((ytest-y(teIdx)).^2);
                
                % R^2
                ssres = sum((ytest-y(teIdx)).^2);
                sstot = sum((y(teIdx)-mean(y(trIdx))).^2);
                r2 = 1-(ssres/sstot);
                r2s(j) = r2;
            end
            mse = mean(mses);
            nmse = mse/mse0;
            r2 = mean(r2s);
%             mse = crossval('mse',X,y,'partition',cp,...
%                 'Predfun',@(xtrain,ytrain,xtest) cv_sven(xtrain,ytrain,xtest,t,lambda2)); % perform CV to get a MSE
            accffen(i,:) = [nmse,r2];
            fprintf('\t\taccffen nmse = %f, r^2 = %f\n',accffen(i,1),accffen(i,2));
        end
        
        accuracy_data{k,z} = {};
        accuracy_data{k,z}.acclasso = acclasso;
        accuracy_data{k,z}.accsven = accsven;
        accuracy_data{k,z}.accffen = accffen;
    end
end

fprintf('\n');
for k = 1:length(n_values)
    n = n_values(k);
    for z = 1:length(d_values)
        d = d_values(z);
        acclasso = accuracy_data{k,z}.acclasso;
        accsven = accuracy_data{k,z}.accsven;
        accffen = accuracy_data{k,z}.accffen;
        fprintf('ntimes = %d, frac_nonzero = %f, n = %d, d = %d\n',ntimes,frac_nonzero,n,d);
        % NMSE
        fprintf('acclasso nmse: %f, %f\n',mean(acclasso(:,1)),std(acclasso(:,1)));
        fprintf('accsven nmse: %f, %f\n',mean(accsven(:,1)),std(accsven(:,1)));
        fprintf('accffen nmse: %f, %f\n',mean(accffen(:,1)),std(accffen(:,1)));
        % R^2
        fprintf('acclasso r^2: %f, %f\n',mean(acclasso(:,2)),std(acclasso(:,2)));
        fprintf('accsven r^2: %f, %f\n',mean(accsven(:,2)),std(accsven(:,2)));
        fprintf('accffen r^2: %f, %f\n',mean(accffen(:,2)),std(accffen(:,2)));
    end
end

if use_parallel
    matlabpool close;
end