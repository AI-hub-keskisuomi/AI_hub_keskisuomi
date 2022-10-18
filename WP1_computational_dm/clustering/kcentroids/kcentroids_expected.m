function [L, C, sumd] = kcentroids_expected(X, k, varargin)
% Description: 
% Perform K-centroids clustering using expected distance estimation strategy
% for treating missing values. Note: City block distance is not supported.   
%
% Function calls:
% [L, C, sumd] = kcentroids_expected(X, k)
% [L, C, sumd] = kcentroids_expected(X, k, replicates)
% [L, C, sumd] = kcentroids_expected(X, k, replicates, distance)
% [L, C, sumd] = kcentroids_expected(X, k, replicates, distance, initcrit)
% [L, C, sumd] = kcentroids_expected(X, k, replicates, distance, initcrit, start)
%
% Inputs:
%          X - Input data set
%          k - Final number of clusters
% replicates - Selected number of repetitions. Default value: 100
%   distance - Selected distance metric. Default value: 'euc' 
%              Alternatives: 
%              'sqe' - squared Euclidean distance
%              'euc' - Euclidean distance
%   initcrit - Initialization criterion. Default value: 'kmeans++' 
%              Alternatives: 
%              'random' - Random selection of initial points
%              'kmeans++' - Kmeans++ based selection of initial points
%      start - Initial values of centroids. The initial number of centroids 
%              can be less or equal to k. Rest of centroids are selected  
%              based on initcrit option if the initial number is less than k.
%              Default value: []  
%
% Output:
%          L - Cluster labels for each observation
%          C - Cluster centroids
%       sumd - Sum of distances
%
defaultReplicates = 100;
defaultDist = 'euc';
defaultInitcrit = 'kmeans++';
defaultStart = [];
%
p = inputParser;
addOptional(p, 'replicates', defaultReplicates, @(x) isnumeric(x));
addOptional(p, 'distance', defaultDist, @(x) ismember(x,{'sqe','euc'}));
addOptional(p, 'initcrit', defaultInitcrit, @(x) ismember(x,{'random','kmeans++'}));
addOptional(p, 'start', defaultStart);
parse(p, varargin{:});
%
replicates = p.Results.replicates;
distance = p.Results.distance;
initcrit = p.Results.initcrit;
start = p.Results.start;
%
sumd = Inf;
C = [];
I = ~all(isnan(X),2);
X = X(I,:);
[Xi, sx] = ecmnmlefunc(X);
for i = 1:replicates
    success = 0;
    while ~success
        [L1, C1, sumd1, success] = clustering(X, k, distance, initcrit, start, Xi, sx);
        if ~success
            if ~isempty(start)
                start(end,:) = [];
            end
        end
    end
    if sumd1 < sumd
        sumd = sumd1;
        L = L1;
        C = C1;
    end
end
% Add zero labels fully incomplete vectors 
L0 = zeros(size(I,1),1);
cnt = 1;
for i = 1:length(I)
    if I(i) == 1
        L0(i) = L(cnt);
        cnt = cnt + 1;
    end
end
L = L0;

end


function [L, C, sumd, success] = clustering(X, k, distance, initcrit, start, Xi, sx)
% Description:
% Perform actual clustering process
%
success = 0;
sumd = Inf;
switch distance
    
    case 'sqe'
    
        X1 = X(~any(isnan(X),2),:);
        if isempty(start), start = X1(randi(size(X1,1)),:); end
        C = start;
        L = sqeucdistfuncp1(C,X1);
        if (strcmp(initcrit,'kmeans++'))
            for i = size(C,1)+1:k
                D = cumsum(sqeucdistfuncp2(C,X1,L));
                if D(end) == 0, C(i:k,:) = X1(ones(1,k-i+1),:); return; end
                C(i,:) = X1(find(rand < D/D(end),1),:);
                L = sqeucdistfuncp1(C,X1);
            end
        elseif (strcmp(initcrit,'random'))
            for i = size(C,1)+1:k
                C(i,:) = X1(randi(size(X1,1)),:);
            end
        end
        % Make sure that all centroids are distinct
        [~, ~, ic] = unique(C, 'rows');
        ic = unique(ic);
        if length(ic) < size(C,1)
            return;
        end        
        L = sqeucdistfuncp1b(C,Xi,sx);
        L1 = 0;
        iter = 0;
        while any(L ~= L1)
            L1 = L;
            for i = 1:k, l = L==i; C(i,:) = meanfun(X(l,:)); end
            if any(any(isnan(C))), return; end
            L = sqeucdistfuncp1b(C,Xi,sx);
            iter = iter + 1;
            if (iter > 250), fprintf('Maximum number of iterations occured\n'), break; end
        end
        D = sqeucdistfuncp2(C,X,L);
        sumd = nansum(D);
        success = 1;        
        
    case 'euc'
        
        X1 = X(~any(isnan(X),2),:);
        if isempty(start), start = X1(randi(size(X1,1)),:); end
        C = start;
        L = eucdistfuncp1(C,X1);
        if (strcmp(initcrit,'kmeans++'))
            for i = size(C,1)+1:k
                D = cumsum(eucdistfuncp2(C,X1,L));
                if D(end) == 0, C(i:k,:) = X1(ones(1,k-i+1),:); return; end
                C(i,:) = X1(find(rand < D/D(end),1),:);
                L = eucdistfuncp1(C,X1);
            end
        elseif (strcmp(initcrit,'random'))
            for i = size(C,1)+1:k
                C(i,:) = X1(randi(size(X1,1)),:);
            end
        end
        % Make sure that all centroids are distinct
        [~, ~, ic] = unique(C, 'rows');
        ic = unique(ic);
        if length(ic) < size(C,1)
            return;
        end
        L = eucdistfuncp1b(C,Xi,sx);
        L1 = 0;
        iter = 0;
        while any(L ~= L1)
            L1 = L;
            for i = 1:k, l = L==i; C(i,:) = spatialmedianfun(Xi(l,:), ...
                                        C(i,:),100,1e-5,sx(l,:),X(l,:)); end              
            if any(any(isnan(C))), return; end
            L = eucdistfuncp1b(C,Xi,sx);
            iter = iter + 1;
            if (iter > 250), fprintf('Maximum number of iterations occured\n'), break; end
        end
        D = eucdistfuncp2(C,X,L);
        sumd = nansum(D);
        success = 1;
        
end

end

function L = sqeucdistfuncp1(C, X) 
% Description: 
% Compute nearest centroids for each observation using squared 
% Euclidean distance. Uses available data strategy for treating missing values. 
%
% Inputs:
%        C - Matrix of cluster centroids
%        X - Input data set
%
% Output:
%        L - Cluster labels for each observation  
%
I = sparse(isnan(X));
X(I) = 0;
[~, L] = min((C.^2)*~I'-2*C*X',[],1);
L = L(:);

end

function L = sqeucdistfuncp1b(C, Xi, sx) 
% Description: 
% Compute nearest centroids for each observation using squared 
% Euclidean distance. Uses expected distance estimation for treating 
% missing values. 
%
% Inputs:
%         C - Matrix of cluster centroids
%        Xi - Inputed data set
%        sx - Variances of original data set  
%
% Output:
%         L - Cluster labels for each observation
%
I = sparse(isnan(C));
I2 = sparse(isnan(Xi));
Xi(I2) = 0;
D = ((Xi.^2)*~I'-2*Xi*C'+~I2*(C.^2)');
D = bsxfun(@plus,D,sum(sx,2));
D(D<eps) = 0;
[~, L] = min(D,[],2);
L = L(:);

end

function D = sqeucdistfuncp2(C, X, L) 
% Description: 
% Compute squared Euclidean distances between observation and centroid 
% matrices. Uses available data strategy for treating missing values. 
%
% Inputs:
%        C - Matrix of cluster centroids
%        X - Input data set
%        L - Cluster labels for each observation
%
% Output:
%        D - Distance matrix  
%
D = bsxfun(@minus,X,C(L,:));
D = nansum(D.^2,2);

end

%{
function D = sqeucdistfuncp2b(C, Xi, L, sx) 
% Description: 
% Compute squared Euclidean distances between observation and 
% centroid matrices. Uses expected distance estimation for treating 
% missing values. 
%
% Inputs:
%        C - Matrix of cluster centroids
%       Xi - Inputed data set
%        L - Cluster labels for each observation
%       sx - Variances of missing values in original X data set
%
% Output:
%        D - Distance matrix   
%
% Calculate omega
omega = bsxfun(@minus,Xi,C(L,:));
omega = sum(omega.^2,2) + sum(sx,2);
% Calculate variance
Ex = Xi;
Ex2 = Xi.^2 + sx;
Ex3 = Xi.^3 + 3*Xi.*sx;
Ex4 = Xi.^4 + 6*(Xi.^2).*sx + 3*sx.^2;
Y = C(L,:);
Ey = Y;
Ey2 = Y.^2;
Ey3 = Y.^3;
Ey4 = Y.^4;
var = sum(Ex4 + Ey4 - 4*Ex3.*Ey - 4*Ex.*Ey3 + 6*Ex2.*Ey2,2) - ...
    sum((Ex2 - 2*Ex.*Ey + Ey2).^2,2);
var(var<0.0000001) = 0;
% Calculate EED
m = (omega.^2)./var;
D = exp(gammaln(m+0.5) - gammaln(m));
D = D.*((omega./m).^(0.5));
ind = isnan(D);
D(ind) = sqrt(omega(ind));

end
%}

function L = eucdistfuncp1(C, X) 
% Description: 
% Compute nearest centroids for each observation using Euclidean 
% distance. Uses available data strategy for treating missing values. 
%
% Inputs:
%         C - Matrix of cluster centroids
%         X - Input data set
%
% Output:
%         L - Cluster labels for each observation
%
I = sparse(isnan(C));
I2 = sparse(isnan(X));
X(I2) = 0;
D = ((X.^2)*~I'-2*X*C'+~I2*(C.^2)');
D(D<eps) = 0;
[~, L] = min(sqrt(D),[],2);
L = L(:);

end

function L = eucdistfuncp1b(C, Xi, sx)
% Description: 
% Compute nearest centroids for each observation using Euclidean 
% distance. Uses expected distance estimation for treating missing values. 
%
% Inputs:
%         C - Matrix of cluster centroids
%        Xi - Inputed data set
%        sx - Variances of missing values in original data X  
%
% Output:
%         L - Cluster labels for each observation
%
I = sparse(isnan(C));
I2 = sparse(isnan(Xi));
Xi(I2) = 0;
D = ((Xi.^2)*~I'-2*Xi*C'+~I2*(C.^2)');
D = bsxfun(@plus,D,sum(sx,2));
D(D<eps) = 0;
[~, L] = min(sqrt(D),[],2);
L = L(:);

end

function D = eucdistfuncp2(C, X, L) 
% Description: 
% Compute Euclidean distances between observation and centroid 
% matrices. Uses available data strategy for treating missing values. 
%
% Inputs:
%        C - Matrix of cluster centroids
%        X - Input data set
%        L - Cluster labels for each observation
%
% Output:
%        D - Distance matrix   
D = bsxfun(@minus,X,C(L,:));
D = sqrt(nansum(D.^2,2));

end

function D = eucdistfuncp2b(C, Xi, L, sx) 
% Description: 
% Compute Euclidean distances between observation and centroid 
% matrices. Uses expected distance estimation for treating missing values. 
%
% Inputs:
%        C - Matrix of cluster centroids
%       Xi - Inputed data set
%        L - Cluster labels for each observation
%       sx - Variances of missing values in original data set X
%
% Output:
%        D - Distance matrix  
%
sx = sum(sx,2);
D = bsxfun(@minus,Xi,C(L,:));
D = sqrt(sum(D.^2,2) + sx);

end

function C = meanfun(X)
% Description: 
% Compute mean value of input data set. 
%
% Input:
%         X - Input data set
%
% Output:
%         C - Mean value of data set  
%
nan = isnan(X);
X(nan) = 0;
C(1,:) = sum(X,1) ./ sum(~nan,1);

end

function C = spatialmedianfun(X, u, max_iter, tol, sx, X_org)
% Description: 
% Compute spatial median value of input data set. Uses expected 
% distance estimation for treating missing values.
%
% Inputs:
%         X - Inputed data set
%         u - Previous spatial median of data
%  max_iter - Maximum number of iterations
%       tol - Tolerance value of convergence
%        sx - Variances of missing values in X_org
%     X_org - Original data set consisting missing values
%
% Output:
%         C - Spatial median value of data set
%
P = ~isnan(X_org);
iters = 0;
w = 1.5;
X_org(isnan(X_org)) = 0;
while iters < max_iter
    iters = iters + 1;
    D = eucdistfuncp2b(u,X,ones(size(X,1),1),sx).^2;
    a = 1./sqrt(D+sqrt(eps));
    a = bsxfun(@times,P,a);
    ax = sum(bsxfun(@times,a,X_org));
    v = (1./sum(a)).*ax;
    u1 = u + w*(v-u);
    if norm(u1-u,inf) < tol
        break;
    end
    u = u1;
end
C = u1;

end

