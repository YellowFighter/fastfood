%Load dataset
[X_protein, y_protein] = load_protein_dataset('/home/kopels/data/CASP.csv');
y_protein_scaled = zscore(y_protein);%2*(y_protein-min(y_protein))/(max(y_protein)-min(y_protein))-1;
X_protein_scaled = zscore(X_protein);

%Set up train and test
random_order = randperm(length(y_protein));
train_inxs = random_order(1:30000);
test_inxs = random_order(30001:end);

%Lasso
lambdas = 0.1:0.05:0.9;
alphas = 0.1:0.05:0.9;
%train
lasso_results = run_lasso(X_protein_scaled(train_inxs,:),y_protein_scaled(train_inxs),{},lambdas,alphas,options);
%test
yfit_lasso = X_protein_scaled(test_inxs,:) * lasso_results.Bfit;
ssres_lasso = sum((yfit_lasso-y_protein_scaled(test_inxs,:)).^2);
sstot_lasso = sum((y_protein_scaled(test_inxs,:)-mean(y_protein_scaled)).^2);
r2_lasso = 1-(ssres_lasso/sstot_lasso);


%FFEN
sigmas = [1,5,10];
use_spiral = 0;
Ns = [1,10,20]*size(X_protein_scaled(train_inxs,:),2);
%train
ffen_results = run_ffen(X_protein_scaled(train_inxs,:),y_protein_scaled(train_inxs),{},.1,.1,sigmas,Ns,use_spiral,options);
rho = corr(X_protein_scaled(train_inxs,:),ffen_results.phi);
Bfitorig = rho*ffen_results.Bfit;

%test
phi = FastfoodForKernel(X_protein_scaled(test_inxs,:)',ffen_results.best_para,ffen_results.best_sig,use_spiral)'; % calculate the projections of the training samples
yfit_ffen = phi*ffen_results.Bfit;
ssres_ffen = sum((yfit_ffen-y_protein_scaled(test_inxs,:)).^2);
sstot_ffen = sum((y_protein_scaled(test_inxs,:)-mean(y_protein_scaled)).^2);
r2_ffen = 1-(ssres_ffen/sstot_ffen);

r2s = [r2_ffen, r2_lasso];
results = [r2s]

%%Start loop%%
num_runs = 20;

for i = 1:num_runs
    %Set up train and test
    random_order = randperm(length(y_protein));
    train_inxs = random_order(1:30000);
    test_inxs = random_order(30001:end);
    
    %Lasso
    lambdas = 0.1:0.05:0.9;
    alphas = 0.1:0.05:0.9;
    %train
    lasso_results = run_lasso(X_protein_scaled(train_inxs,:),y_protein_scaled(train_inxs),{},lambdas,alphas,options);
    %test
    yfit_lasso = X_protein_scaled(test_inxs,:) * lasso_results.Bfit;
    ssres_lasso = sum((yfit_lasso-y_protein_scaled(test_inxs,:)).^2);
    sstot_lasso = sum((y_protein_scaled(test_inxs,:)-mean(y_protein_scaled)).^2);
    r2_lasso = 1-(ssres_lasso/sstot_lasso);
    
    
    %FFEN
    sigmas = [1,5,10];
    use_spiral = 0;
    Ns = [1,10,20]*size(X_protein_scaled(train_inxs,:),2);
    %train
    ffen_results = run_ffen(X_protein_scaled(train_inxs,:),y_protein_scaled(train_inxs),{},.1,.1,sigmas,Ns,use_spiral,options);
    rho = corr(X_protein_scaled(train_inxs,:),ffen_results.phi);
    Bfitorig = rho*ffen_results.Bfit;
    
    %test
    phi = FastfoodForKernel(X_protein_scaled(test_inxs,:)',ffen_results.best_para,ffen_results.best_sig,use_spiral)'; % calculate the projections of the training samples
    yfit_ffen = phi*ffen_results.Bfit;
    ssres_ffen = sum((yfit_ffen-y_protein_scaled(test_inxs,:)).^2);
    sstot_ffen = sum((y_protein_scaled(test_inxs,:)-mean(y_protein_scaled)).^2);
    r2_ffen = 1-(ssres_ffen/sstot_ffen);
   
    r2s = [r2_ffen, r2_lasso];
    results = vertcat(results,r2s);
    i/num_runs
end

results