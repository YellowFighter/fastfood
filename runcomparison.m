function [ data_out ] = runcomparison( X,y,options,ntimes,cmpname,cvpart,d,n)
mse0 = (1/size(y,1))*sum((y-mean(y)).^2);

try % test whether we can use Spiral package (c impl of FWHT)
    fwht_spiral([1; 1]);
    use_spiral = 1
catch
    use_spiral = 0
end

dbgmsg('Running %s comparison.',cmpname);
data_out = {};
for k = 1:ntimes
    dbgmsg('(%d) Starting iter %d',k,k);

    %% (Hyper-)params
    lambda2 = 0.1;
    alpha = 0.5;
    t = lambda2*alpha;
    N = d*20; % number of basis functions to use for approximation
    para = FastfoodPara(N,d); % generate FF parameters
    sigma = 10; % band-width of Gaussian kernel
    cp = cvpart(n);

    %% Built-in lin regresison using lasso
    accslassoreg = []; % accuracies
    timslassoreg = []; % times
    for l=1:cp.NumTestSets
        dbgmsg('(%d) Linear LASSO l=%d',k,l);
        trIdx = cp.training(l);
        teIdx = cp.test(l);
        tic;
        ytest = cv_lasso(X(trIdx,:),y(trIdx),X(teIdx,:),1,1,options); %Use plain old linear reg here
        timlassoreg = toc;
        mse = 1/length(ytest)*sum((ytest-y(teIdx)).^2);
        ssres = sum((ytest-y(teIdx)).^2);
        sstot = sum((y(teIdx)-mean(y(trIdx))).^2);
        r2 = 1-(ssres/sstot);
        nmse = mse/mse0;
        acclassoreg = [nmse,r2];
        accslassoreg(l,:) = acclassoreg;
        timslassoreg(l) = timlassoreg;
    end
    
    %% Built-in lasso
    accslasso = []; % accuracies
    timslasso = []; % times
    for l=1:cp.NumTestSets
        dbgmsg('(%d) Linear LASSO l=%d',k,l);
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
        dbgmsg('(%d) Linear SVEN l=%d',k,l);
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
        dbgmsg('(%d) Linear FFEN l=%d',k,l);
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


    data_out{k} = {};
    data_out{k}.acclasso = accslasso;
    data_out{k}.acclassoreg = accslassoreg;
    data_out{k}.accsven = accssven;
    data_out{k}.accffen = accsffen;
    data_out{k}.timlasso = timslasso;
    data_out{k}.timsven = timssven;
    data_out{k}.timffen = timsffen;
end