%Load dataset
[X_kegg, y_kegg] = load_kegg_dataset('/home/kopels/data/kegg.data',2);
good_inxs = find(~any(isnan(X_kegg),2));
X_kegg = X_kegg(good_inxs,:) ;
y_kegg = y_kegg(good_inxs);
y_kegg_scaled = zscore(y_kegg);%2*(y_kegg-min(y_kegg))/(max(y_kegg)-min(y_kegg))-1;
X_kegg_scaled = zscore(X_kegg);

%Set up train and test
random_order = randperm(length(y_kegg));
train_inxs = random_order(1:50000);
test_inxs = random_order(50001:end);

%Lasso
lambdas = 0.1:0.05:0.9;
alphas = 0.1:0.05:0.9;
%train
lasso_results = run_lasso(X_kegg_scaled(train_inxs,:),y_kegg_scaled(train_inxs),{},lambdas,alphas,options);
%test
yfit_lasso = X_kegg_scaled(test_inxs,:) * lasso_results.Bfit;
ssres_lasso = sum((yfit_lasso-y_kegg_scaled(test_inxs,:)).^2);
sstot_lasso = sum((y_kegg_scaled(test_inxs,:)-mean(y_kegg_scaled)).^2);
r2_lasso = 1-(ssres_lasso/sstot_lasso);


%FFEN
sigmas = [1,5,10];
use_spiral = 0;
Ns = [1,10,20]*size(X_kegg_scaled(train_inxs,:),2);
%train
ffen_results = run_ffen(X_kegg_scaled(train_inxs,:),y_kegg_scaled(train_inxs),{},.1,.1,sigmas,Ns,use_spiral,options);
rho = corr(X_kegg_scaled(train_inxs,:),ffen_results.phi);
Bfitorig = rho*ffen_results.Bfit;

%test
phi = FastfoodForKernel(X_kegg_scaled(test_inxs,:)',ffen_results.best_para,ffen_results.best_sig,use_spiral)'; % calculate the projections of the training samples
yfit_ffen = phi*ffen_results.Bfit;
ssres_ffen = sum((yfit_ffen-y_kegg_scaled(test_inxs,:)).^2);
sstot_ffen = sum((y_kegg_scaled(test_inxs,:)-mean(y_kegg_scaled)).^2);
r2_ffen = 1-(ssres_ffen/sstot_ffen);

r2s = [r2_ffen, r2_lasso];
results = [r2s]

%%Start loop%%
num_runs = 20;

for i = 1:num_runs
    %Set up train and test
    random_order = randperm(length(y_kegg));
    train_inxs = random_order(1:50000);
    test_inxs = random_order(50001:end);
    
    %Lasso
    lambdas = 0.1:0.05:0.9;
    alphas = 0.1:0.05:0.9;
    %train
    lasso_results = run_lasso(X_kegg_scaled(train_inxs,:),y_kegg_scaled(train_inxs),{},lambdas,alphas,options);
    %test
    yfit_lasso = X_kegg_scaled(test_inxs,:) * lasso_results.Bfit;
    ssres_lasso = sum((yfit_lasso-y_kegg_scaled(test_inxs,:)).^2);
    sstot_lasso = sum((y_kegg_scaled(test_inxs,:)-mean(y_kegg_scaled)).^2);
    r2_lasso = 1-(ssres_lasso/sstot_lasso);
    
    
    %FFEN
    sigmas = [1,5,10];
    use_spiral = 0;
    Ns = [1,10,20]*size(X_kegg_scaled(train_inxs,:),2);
    %train
    ffen_results = run_ffen(X_kegg_scaled(train_inxs,:),y_kegg_scaled(train_inxs),{},.1,.1,sigmas,Ns,use_spiral,options);
    rho = corr(X_kegg_scaled(train_inxs,:),ffen_results.phi);
    Bfitorig = rho*ffen_results.Bfit;
    
    %test
    phi = FastfoodForKernel(X_kegg_scaled(test_inxs,:)',ffen_results.best_para,ffen_results.best_sig,use_spiral)'; % calculate the projections of the training samples
    yfit_ffen = phi*ffen_results.Bfit;
    ssres_ffen = sum((yfit_ffen-y_kegg_scaled(test_inxs,:)).^2);
    sstot_ffen = sum((y_kegg_scaled(test_inxs,:)-mean(y_kegg_scaled)).^2);
    r2_ffen = 1-(ssres_ffen/sstot_ffen);
   
    r2s = [r2_ffen, r2_lasso];
    results = vertcat(results,r2s);
    i/num_runs
end

results
