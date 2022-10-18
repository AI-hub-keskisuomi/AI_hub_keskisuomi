function C = nanspatialmedianfunc(X, varargin)
% Description: 
% Estimate spatial median value of input data set. Algorithm uses 
% available data strategy for treating missing values.  
%
% Function calls:
%         C = nanspatialmedianfunc(X)
%         C = nanspatialmedianfunc(X, max_iter)
%         C = nanspatialmedianfunc(X, max_iter, tol)  
%
% Inputs: 
%         X - Input data set
%  max_iter - Maximum number of iterations 
%             Default value: 100 
%       tol - Stopping criterion 
%             Default value: 1e-5 
%
% Output: 
%         C - Spatial median value of data set 
%
defaultMaxIter = 100;
defaultTol = 1e-5; 
%
p = inputParser;
addOptional(p, 'max_iter', defaultMaxIter, @(x) isnumeric(x));
addOptional(p, 'tol', defaultTol, @(x) isnumeric(x));
parse(p,varargin{:});
max_iter = p.Results.max_iter;
tol = p.Results.tol;
%
u = zeros(1,size(X,2));
for i = 1:size(X,2)
    I = ~isnan(X(:,i));
    u(i) = median(X(I,i));
end
P = ~isnan(X);
iters = 0;
w = 1.5;
while iters < max_iter
    iters = iters + 1;
    D = sqrt(nansum(bsxfun(@minus,X,u).^2,2));
    a = 1./(D+sqrt(sqrt(eps)));
    ax = nansum(bsxfun(@times,a,X));
    a = bsxfun(@times,P,a);
    v = (1./sum(a)).*ax;
    u1 = u + w*(v-u);
    if norm(u1-u,inf) < tol
        break;
    end
    u = u1;
end
C = u1;

end

