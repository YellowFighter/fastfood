function results = run_ffen(X,y,cp,lambdas,alphas,sigs,Ns,use_spiral,options)
% lambdas - values of lambda to try
% alphas - values of alpha to try
% cp is cross-validation structure. Set to {} if you don't want to do CV.
% sig - band-width of Gaussian kernel
% N - number of basis functions to use for approximation


%% Built-in lasso
accsffen = []; % accuracies
timsffen = []; % times

d = size(X,2);
para = FastfoodPara(Ns(1),d); % generate FF parameters
% currently broken
if isempty(cp) == 0
    for l=1:cp.NumTestSets
        dbgmsg('(%d) FFEN l=%d',k,l);
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
                ytest = cv_ffen(X(intTrIdx,:),y(intTrIdx),X(intTeIdx,:),alpha,lambda,para,sig,use_spiral,options);
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
        ytest = cv_ffen(X(trIdx,:),y(trIdx),X(teIdx,:),best_alpha,best_lambda,para,sig,use_spiral,options);
        timlasso = toc;
        mse = 1/length(ytest)*sum((ytest-y(teIdx)).^2);
        ssres = sum((ytest-y(teIdx)).^2);
        sstot = sum((y(teIdx)-mean(y(trIdx))).^2);
        r2 = 1-(ssres/sstot);
        nmse = mse/mse0;
        acclasso = [nmse,r2];
        accsffen(l,:) = accffen;
        timsffen(l) = timffen;
    end
end

results.accsffen = accsffen;
results.timsffen = timsffen;

% Now run for all of the data
best_r2 = -Inf;
% Now we need to divide up into training and testing again,
% but we will do a very simple percentage to keep things
% fast
fracTest = 0.5;
numTest = round(size(X,1))*fracTest;
inxs = randperm(size(X,1));
testInxs = inxs(1:numTest);
trainInxs = inxs((numTest+1):end);
intTrIdx = 1:size(X,1);%trainInxs;
intTeIdx = 1:size(X,1);%testInxs;
for alpha=alphas
    for lambda=lambdas
        for sig=sigs
            for N=Ns
                para = FastfoodPara(N,d); % generate FF parameters
                ytest = cv_ffen(X(intTrIdx,:),y(intTrIdx),X(intTeIdx,:),alpha,lambda,para,sig,use_spiral,options);
                ssres = sum((ytest-y(intTeIdx)).^2);
                sstot = sum((y(intTeIdx)-mean(y(intTrIdx))).^2);
                r2 = 1-(ssres/sstot);
                if (r2 > best_r2)
                    best_alpha = alpha;
                    best_lambda = lambda;
                    best_r2 = r2;
                    best_N = N;
                    best_sig = sig;
                    best_para = para;
                end
            end
        end
    end
end

%para = FastfoodPara(best_N,d); % generate FF parameters
phi = FastfoodForKernel(X',best_para,best_sig,use_spiral)'; % calculate the projections of the training samples
Bfit = lasso(phi,y,'alpha',best_alpha,'lambda',best_lambda,'options',options); % perform LASSO on the projections to learn regression coefficients
% if options.UseParallel
%     phitest = gpuArray(phitest);
% end
yfit = phi*Bfit; % perform regression

ssres = sum((yfit-y).^2);
sstot = sum((y-mean(y)).^2);
r2 = 1-(ssres/sstot);

results.phi = phi;
results.Bfit = Bfit;
results.yfit = yfit;
results.R2fit = r2;
results.best_para = best_para;
results.best_sig = best_sig;
results.best_alpha = best_alpha;
results.best_lambda = best_lambda;

% Now convert back to original space
rho = corr(X,phi);
Bfitorig = rho*Bfit;
Bfitorig = Bfitorig/norm(Bfitorig);%*norm(B_lin);

