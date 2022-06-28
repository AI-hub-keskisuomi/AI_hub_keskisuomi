function [Xi, sx, success] = ecmnmlefunc(X)
% Description: 
% Computes conditional means and variances for data with missing values.
%
% Function call:
% [Xi, sx] = ecmnmlefunc(X)
%
% Input:
%         X - Original data set with missing values 
%
% Outputs:
%        Xi - Imputed data set by conditional means  
%        sx - Conditional variances of missing values
%   success - Indicates if algorithm successfully finished  
%
success = 1;
warning off;
if sum(isnan(X(:))) > 0
    [Nx, nx] = size(X);
    [Mu, Covariance, success] = EM(X);
    if ~success
        Xi = X;
        sx = zeros(size(X));
        warning on;
        return;
    end
    sx = zeros(Nx,nx);
    for i = 1:Nx
        [~, missing] = find(isnan(X(i,:)));
        [~, notmissing] = find(~isnan(X(i,:)));
        if length(missing) ~= nx && length(notmissing) ~= nx
            Cov = getcovariances(Covariance,missing,notmissing);
            mu1 = zeros(nx,1);
            mu1(missing) = Mu(missing) + Cov{2}*(Cov{4}^-1)* ...
                                    (X(i,notmissing)'-Mu(notmissing));
            X(i,missing) = mu1(missing);
            cov1 = Cov{1} - Cov{2}*(Cov{4}^-1)*Cov{3};
            sx(i,missing) = diag(cov1);
        end
    end
else
    sx = zeros(size(X));
end
Xi = X;
warning on;

end

function [mu, sigma, success] = EM(Xm)
%
% Description: 
% Estimate conditional mean vector and covariance matrix based on expectation 
% maximization algorithm. Missing values are conditioned by available values. 
% Maximum negative log-likelihood value is used as the stopping criterion 
% of iterative algorithm.
%
% Input:
%          Xm - Input data set with missing values
%
% Outputs:
%          mu - Conditional mean vector
%       sigma - Conditional covariance matrix
%     success - Indicates if algorithm successfully finished  
%
success = 1;
Xi = Xm;
mu = mean(Xm,'omitnan');
for i = 1:size(Xm,2)
    I = isnan(Xm(:,i));
    Xi(I,i) = mu(i);
end
mu = mean(Xi);
sigma = cov(Xi,1);

r1 = Inf;
iters = 1;
max_iter = 200;
tol = 1e-5;
[N, n] = size(Xm);
while iters < max_iter
    r = r1;
    iters = iters + 1;
    B = zeros(n,n);
    for j = 1:N
        X1 = Xm(j,:);
        mi1 = find(isnan(X1));
        av1 = find(~isnan(X1));
        if length(av1) < n
            B(mi1,mi1) = B(mi1,mi1) + sigma(mi1,mi1) - ...
                        sigma(mi1,av1)/(sigma(av1,av1))*sigma(av1,mi1);
            Xi(j,mi1) = mu(mi1)' + ... 
                        sigma(mi1,av1)/(sigma(av1,av1))*(Xi(j,av1)-mu(av1))';
        end
    end
    mu = mean(Xi);
    sigma = cov(Xi,1) + B./N;
    
    % Convergence check
    [U, fail] = chol(sigma);
    if fail 
        % Not positive definite matrix
        success = 0;
        break;
    end
    y = bsxfun(@minus,Xi,mu);
    r1 = sum((1/2)*(2*sum(log(diag(U))) + diag(y/sigma*y') + n*log(2*pi)));
    if norm(r1-r,inf) < tol
        break;
    end
end
if (iters == max_iter)
    fprintf('Maximum number of iterations reached in EM algorithm!\n');
end
mu = mu';

end

%{
function [s, sigma] = ADS(Xm)
%
% Description: 
% Uses available data strategy for estimating statistical parameters of Gaussian 
% distribution. 
%
% Input:
%         Xm - Input data set with missing values
%
% Outputs:
%          s - Spatial median of data set
%      sigma - Covariance matrix of data set
%
[N, ~] = size(Xm);
s = nanspatialmedianfunc(Xm);
P = ~isnan(Xm);
Xm(isnan(Xm)) = 0;
sigma = (P.*(Xm-s))'*(P.*(Xm-s))./(N-1);
s = s';

end
%}

function covariances = getcovariances(M, missing, notmissing)
%
% Description: 
% Divides covariance matrix into the main parts.
%
% Inputs:
%          M - Original covariance matrix
%    missing - Missing components of data vector
% notmissing - Available components of data vector
%
% Output:
% covariances - Partitioned covariance matrix
%
Cov11 = M;
Cov11(notmissing,:) = [];
Cov11(:,notmissing) = [];
Cov12 = M;
Cov12(notmissing,:) = [];
Cov12(:,missing) = [];
Cov21 = M;
Cov21(missing,:) = [];
Cov21(:,notmissing) = [];
Cov22 = M;
Cov22(missing,:) = [];
Cov22(:,missing) = [];
covariances = cell(4,1);
covariances{1} = Cov11;
covariances{2} = Cov12;
covariances{3} = Cov21;
covariances{4} = Cov22;

end

