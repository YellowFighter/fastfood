%setup
dbgmsg('Starting');
global DEBUG
DEBUG = true;
rng(1);
warning('off','all');
use_parallel = false;
options = statset('UseParallel',use_parallel);


% load dataset
[x_music_test,y_music_test, x_music_train, y_music_train] = load_music('/home/kopels/data/YearPredictionMSD.txt')

%FFEN
sigmas = [1,5,10];
use_spiral = 0;
Ns = [1,10,20]*size(x_music_train,2);
%train
ffen_results = run_ffen(x_music_train,y_music_train,{},.1,.1,sigmas,Ns,use_spiral,options);
rho = corr(x_music_train,ffen_results.phi);
Bfitorig = rho*ffen_results.Bfit;

%test
phi = FastfoodForKernel(x_music_test',ffen_results.best_para,ffen_results.best_sig,use_spiral)'; % calculate the projections of the training samples
yfit_ffen = phi*ffen_results.Bfit;
ssres_ffen = sum((yfit_ffen-y_music_test).^2);
sstot_ffen = sum((y_music_test-mean(y_music_test)).^2);
r2_ffen = 1-(ssres_ffen/sstot_ffen)


%Lasso
lambdas = 0.1:0.05:0.9;
alphas = 0.1:0.05:0.9;
%train
lasso_results = run_lasso(x_music_train,y_music_train,{},lambdas,alphas,options);
%test
yfit_lasso = x_music_test * lasso_results.Bfit;
ssres_lasso = sum((yfit_lasso-y_music_test).^2);
sstot_lasso = sum((y_music_test-mean(y_music_test)).^2);
r2_lasso = 1-(ssres_lasso/sstot_lasso)