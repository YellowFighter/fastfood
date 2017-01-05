function [ data_out ] = runcomparison( X,y,options,ntimes,cmpname,cvpart,d)
mse0 = (1/size(y,1))*sum((y-mean(y)).^2);
lambdas = linspace(0.01,0.99,50);
alphas = linspace(0.01,0.99,50);

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
    cp = cvpart(size(X,1));
    
    %% Built-in lin regresison
    accsreg = []; % accuracies
    timsreg = []; % times
    for l=1:cp.NumTestSets
        dbgmsg('(%d) Linear LASSO l=%d',k,l);
        trIdx = cp.training(l);
        teIdx = cp.test(l);
        tic;
        %ytest = cv_lasso(X(trIdx,:),y(trIdx),X(teIdx,:),1,1,options);
        B = regress(y(trIdx),X(trIdx,:)); %Using linear regression
        ytest = X(teIdx,:)*B;
        timreg = toc;
        mse = 1/length(ytest)*sum((ytest-y(teIdx)).^2);
        ssres = sum((ytest-y(teIdx)).^2);
        sstot = sum((y(teIdx)-mean(y(trIdx))).^2);
        r2 = 1-(ssres/sstot);
        nmse = mse/mse0;
        accreg = [nmse,r2];
        accsreg(l,:) = accreg;
        timsreg(l) = timreg;
    end
    
    %% Built-in lasso
    accslasso = []; % accuracies
    timslasso = []; % times
    for l=1:cp.NumTestSets
        dbgmsg('(%d) Linear LASSO l=%d',k,l);
        trIdx = cp.training(l);
        teIdx = cp.test(l);
        tic;
        best_r2 = -Inf;
        % Now we need to divide up into training and testing again,
        % but we will do a very simple percentage to keep things
        % fast
        fracTest = 0.3;
        numTest = round(length(trIdx))*fracTest;
        inxs = randperm(length(trIdx));
        testInxs = inxs(1:numTest);
        trainInxs = inxs((numTest+1):end);
        intTrIdx = trIdx(trainInxs);
        intTeIdx = trIdx(testInxs);
        for alpha=alphas
            for lambda=lambdas
                ytest = cv_lasso(X(intTrIdx,:),y(intTrIdx),X(intTeIdx,:),alpha,lambda,options);
                ssres = sum((ytest-y(intTeIdx)).^2);
                sstot = sum((y(intTeIdx)-mean(y(intTrIdx))).^2);
                r2 = 1-(ssres/sstot);
                if (r2 > best_r2)
                    best_alpha = alpha;
                    best_lambda = lambda;
                    best_r2 = r2;
                end
            end
        end
        ytest = cv_lasso(X(trIdx,:),y(trIdx),X(teIdx,:),best_alpha,best_lambda,options);
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
    
%     %% SVEN
%     accssven = []; % accuracies
%     timssven = []; % times
%     for l=1:cp.NumTestSets
%         dbgmsg('(%d) Linear SVEN l=%d',k,l);
%         trIdx = cp.training(l);
%         teIdx = cp.test(l);
%         tic;
%         best_alpha = .1;
%         best_lambda = .1;
%         best_r2 = -Inf;
%         % Now we need to divide up into training and testing again,
%         % but we will do a very simple percentage to keep things
%         % fast
%         fracTest = 0.3;
%         numTest = round(length(trIdx))*fracTest;
%         inxs = randperm(length(trIdx));
%         testInxs = inxs(1:numTest);
%         trainInxs = inxs((numTest+1):end);
%         intTrIdx = trIdx(trainInxs);
%         intTeIdx = trIdx(testInxs);
%         for alpha= .1:.01:.8
%             for lambda= .1:.01:.8
%                 t = lambda*alpha;
%                 ytest = cv_sven(X(intTrIdx,:),y(intTrIdx),X(intTeIdx,:),t,lambda,options);
%                 ssres = sum((ytest-y(intTeIdx)).^2);
%                 sstot = sum((y(intTeIdx)-mean(y(intTrIdx))).^2);
%                 r2 = 1-(ssres/sstot);
%                 if (r2 > best_r2)
%                     best_alpha = alpha;
%                     best_lambda = lambda;
%                     best_r2 = r2;
%                 end
%             end
%         end
%         t = best_lambda*best_alpha;
%         ytest = cv_sven(X(trIdx,:),y(trIdx),X(teIdx,:),t,best_lambda,options);
%         timsven = toc;
%         mse = 1/length(ytest)*sum((ytest-y(teIdx)).^2);
%         ssres = sum((ytest-y(teIdx)).^2);
%         sstot = sum((y(teIdx)-mean(y(trIdx))).^2);
%         r2 = 1-(ssres/sstot);
%         nmse = mse/mse0;
%         accsven = [nmse,r2];
%         accssven(l,:) = accsven;
%         timssven(l) = timsven;
%     end
    
    
    %% FFEN
    sigma = 10; % band-width of Gaussian kernel
    N = d*20; % number of basis functions to use for approximation
    para = FastfoodPara(N,d); % generate FF parameters
    accsffen = []; % accuracies
    timsffen = []; % times
    for l=1:cp.NumTestSets
        dbgmsg('(%d) Linear FFEN l=%d\n',k,l);
        trIdx = cp.training(l);
        teIdx = cp.test(l);
        % Now we need to divide up into training and testing again,
        % but we will do a very simple percentage to keep things
        % fast
        best_r2 = -Inf;
        fracTest = 0.3;
        numTest = round(length(trIdx))*fracTest;
        inxs = randperm(length(trIdx));
        testInxs = inxs(1:numTest);
        trainInxs = inxs((numTest+1):end);
        intTrIdx = trIdx(trainInxs);
        intTeIdx = trIdx(testInxs);
        for alpha=alphas
            for lambda=lambdas
                ytest = cv_ffen(X(intTrIdx,:),y(intTrIdx),X(intTeIdx,:),alpha,lambda,para,sigma,use_spiral,options);
                ssres = sum((ytest-y(intTeIdx)).^2);
                sstot = sum((y(intTeIdx)-mean(y(intTrIdx))).^2);
                r2 = 1-(ssres/sstot);
                if (r2 > best_r2)
                    best_alpha = alpha;
                    best_lambda = lambda;
                    best_r2 = r2;
                end
            end
        end
        tic;
        ytest = cv_ffen(X(trIdx,:),y(trIdx),X(teIdx,:),best_alpha,best_lambda,para,sigma,use_spiral,options);
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
    data_out{k}.accreg = accsreg;
    %data_out{k}.accsven = accssven;
    data_out{k}.accffen = accsffen;
    data_out{k}.timlasso = timslasso;
    %data_out{k}.timsven = timssven;
    data_out{k}.timffen = timsffen;
end