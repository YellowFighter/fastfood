%Load dataset
[X_park, y_park] = load_park_dataset('/home/kopels/data/parkinsons_updrs.data');
y_park_scaled = zscore(y_park);%2*(y_park-min(y_park))/(max(y_park)-min(y_park))-1;
X_park_scaled = zscore(X_park);

%Set up train and test
random_order = randperm(length(y_park));
train_inxs = random_order(1:2500);
test_inxs = random_order(2501:end);

%Lasso
lambdas = 0.1:0.05:0.9;
alphas = 0.1:0.05:0.9;
%train
lasso_results = run_lasso(X_park_scaled(train_inxs,:),y_park_scaled(train_inxs),{},lambdas,alphas,options);
%test
yfit_lasso = X_park_scaled(test_inxs,:) * lasso_results.Bfit;
ssres_lasso = sum((yfit_lasso-y_park_scaled(test_inxs,:)).^2);
sstot_lasso = sum((y_park_scaled(test_inxs,:)-mean(y_park_scaled)).^2);
r2_lasso = 1-(ssres_lasso/sstot_lasso)


%FFEN
sigmas = [1,5,10];
use_spiral = 0;
Ns = [1,10,20]*size(X_park_scaled(train_inxs,:),2);
%train
ffen_results = run_ffen(X_park_scaled(train_inxs,:),y_park_scaled(train_inxs),{},.1,.1,sigmas,Ns,use_spiral,options);
rho = corr(X_park_scaled(train_inxs,:),ffen_results.phi);
Bfitorig = rho*ffen_results.Bfit;

%test
phi = FastfoodForKernel(X_park_scaled(test_inxs,:)',ffen_results.best_para,ffen_results.best_sig,use_spiral)'; % calculate the projections of the training samples
yfit_ffen = phi*ffen_results.Bfit;
ssres_ffen = sum((yfit_ffen-y_park_scaled(test_inxs,:)).^2);
sstot_ffen = sum((y_park_scaled(test_inxs,:)-mean(y_park_scaled)).^2);
r2_ffen = 1-(ssres_ffen/sstot_ffen)



