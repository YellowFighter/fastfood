%% Generate/load data
n = 1000;
d = 150;
X = randn(n,d);
r = zeros(size(X,2),1);
r(1:5)= [0;3;0;-3;0]; % only two nonzero coefficients (2,4)
y = X*r + randn(size(X,1),1)*.1; % small added noise
% mean center and unit variance

%% Hyper params
lambda2 = 0.1;
alpha = 0.5;
t = lambda2*alpha;

%% Built-in lasso
tic;
B = lasso(X,y,'alpha',alpha,'lambda',lambda2);
tlasso = toc;
find(B(:,1) ~= 0)

%% SVEN
tic;
beta = SVEN(X',y',t,lambda2);
tsven = toc;
find(beta ~= 0)

%% FFEN
try
    % test whether we can use Spiral package
    fwht_spiral([1; 1]);
    use_spiral = 1;
catch
    display('Cannot perform Walsh-Hadamard transform using Spiral WHT package.');
    display('Use Matlab function fwht instead, which is slow for large-scale data.')
    use_spiral = 0;
end
N = d*20; % number of basis functions to use for approximation
para = FastfoodPara(N,d);
sigma = 10; % band-width of Gaussian kernel
tic;
phi = FastfoodForKernel(X',sigma,para,use_spiral)';
tphi = toc;
tic;
B = lasso(phi,Y);
tffen = toc;