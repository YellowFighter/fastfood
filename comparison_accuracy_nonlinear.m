%rng(1);
warning('off','all');
try % test whether we can use Spiral package
    fwht_spiral([1; 1]);
    use_spiral = 1;
catch
    use_spiral = 0;
end

ntimes = 1;
frac_nonzero = 0.1;

n_values = [10000,100000];
d_values = [600,1000,10000,100000];
accuracy_data = {};

%y_func = @(X,r) X*r + randn(size(X,1),1)*.1; % linear function
y_func = @(X,r,shuffle) (X.*X(:,shuffle))*r + randn(size(X,1),1)*.1;

for k = 1:length(n_values)
    n = n_values(k);
    for z = 1:length(d_values)
        d = d_values(z);
        fprintf('data n = %d, d = %d\n',n,d);
        
        shuffle = randperm(d);
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
        y = y_func(X,r,shuffle);
        X = zscore(X); % mean center and unit variance
        y = zscore(y); % mean center and unit variance
        mse0 = (1/size(y,1))*sum((y-mean(y)).^2);

        %% (Hyper-)params
        lambda2 = 0.1;
        alpha = 0.5;
        t = lambda2*alpha;
        N = d*20; % number of basis functions to use for approximation
        para = FastfoodPara(N,d); % generate FF parameters
        cp = cvpartition(n,'holdout',0.3); % hold out 30% for testing
        sigma = 10; % band-width of Gaussian kernel

        %% Built-in lasso
        trIdx = cp.training;
        teIdx = cp.test;
        ytest = cv_lasso(X(trIdx,:),y(trIdx),X(teIdx,:),alpha,lambda2);
        mse = 1/length(ytest)*sum((ytest-y(teIdx)).^2);
        ssres = sum((ytest-y(teIdx)).^2);
        sstot = sum((y(teIdx)-mean(y(trIdx))).^2);
        r2 = 1-(ssres/sstot);
        nmse = mse/mse0;
        acclasso = [nmse,r2];
        fprintf('\tacclasso nmse = %f, r^2 = %f\n',acclasso(1),acclasso(2));

        %% SVEN
        trIdx = cp.training;
        teIdx = cp.test;
        ytest = cv_sven(X(trIdx,:),y(trIdx),X(teIdx,:),t,lambda2);
        mse = 1/length(ytest)*sum((ytest-y(teIdx)).^2);
        ssres = sum((ytest-y(teIdx)).^2);
        sstot = sum((y(teIdx)-mean(y(trIdx))).^2);
        r2 = 1-(ssres/sstot);
        nmse = mse/mse0;
        accsven = [nmse,r2];
        fprintf('\taccsven nmse = %f, r^2 = %f\n',accsven(1),accsven(2));


        %% FFEN
        trIdx = cp.training;
        teIdx = cp.test;
        ytest = cv_ffen(X(trIdx,:),y(trIdx),X(teIdx,:),alpha,lambda2,para,sigma,use_spiral);
        mse = 1/length(ytest)*sum((ytest-y(teIdx)).^2);
        ssres = sum((ytest-y(teIdx)).^2);
        sstot = sum((y(teIdx)-mean(y(trIdx))).^2);
        r2 = 1-(ssres/sstot);
        nmse = mse/mse0;
        accffen = [nmse,r2];
        fprintf('\taccffen nmse = %f, r^2 = %f\n',accffen(1),accffen(2));

        
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
        accs = [acclasso; accsven; accffen];
        fname = fullfile('results','nonlinear',sprintf('n-%d_d-%d.csv',n,d));
        csvwrite(fname,accs);
    end
end

if use_parallel
    matlabpool close;
end