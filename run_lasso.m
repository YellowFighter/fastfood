function results = run_lasso(X,y,cp,lambdas,alphas,options)
% lambdas - values of lambda to try
% alphas - values of alpha to try
% cp is cross-validation structure. Set to {} if you don't want to do CV.


%% Built-in lasso
accslasso = []; % accuracies
timslasso = []; % times

if isempty(cp) == 0
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
end

results.accslasso = accslasso;
results.timslasso = timslasso;

% Now run for all of the data
best_r2 = -Inf;
% Now we need to divide up into training and testing again,
% but we will do a very simple percentage to keep things
% fast
fracTest = 0.3;
numTest = round(size(X,1))*fracTest;
inxs = randperm(size(X,1));
testInxs = inxs(1:numTest);
trainInxs = inxs((numTest+1):end);
intTrIdx = trainInxs;
intTeIdx = testInxs;
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


Bfit = lasso(X,y,'alpha',best_alpha,'lambda',best_lambda,'options',options); % perform LASSO on the projections to learn regression coefficients
% if options.UseParallel
%     xtest = gpuArray(xtest);
%     B = gpuArray(B);
% end
yfit = X*Bfit; % perform regression
ssres = sum((yfit-y).^2);
sstot = sum((y-mean(y)).^2);
r2 = 1-(ssres/sstot);
        
results.Bfit = Bfit;
results.yfit = yfit;
results.R2fit = r2;

