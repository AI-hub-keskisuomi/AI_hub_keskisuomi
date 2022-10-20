function [L, C, sumd] = kcentroids(X, k, varargin)
% Description: 
% Perform traditional K-centroids clustering using available data strategy 
% for treating missing values.  
%
% Function calls:
% [L, C, sumd] = kcentroids(X, k)
% [L, C, sumd] = kcentroids(X, k, replicates)
% [L, C, sumd] = kcentroids(X, k, replicates, distance)
% [L, C, sumd] = kcentroids(X, k, replicates, distance, initcrit)
% [L, C, sumd] = kcentroids(X, k, replicates, distance, initcrit, start)
%
% Inputs:
%          X - Input data set 
%          k - Final number of clusters
% replicates - Selected number of repetitions. Default value: 100
%   distance - Selected distance metric. Default value: 'euc' 
%              Alternatives: 
%              'sqe' - squared Euclidean distance
%              'euc' - Euclidean distance
%              'cit' - City block distance 
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
addOptional(p, 'distance', defaultDist, @(x) ismember(x,{'sqe','euc','cit'}));
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
for i = 1:replicates
    success = 0;
    while ~success
        [L1, C1, sumd1, success] = clustering(X, k, distance, initcrit, start);
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

function [L, C, sumd, success] = clustering(X, k, distance, initcrit, start)
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
        L = sqeucdistfuncp1(C,X);
        L1 = 0;
        iter = 0;
        while any(L ~= L1)
            L1 = L;
            for i = 1:k, l = L==i; C(i,:) = meanfun(X(l,:)); end
            if any(any(isnan(C))), return; end
            L = sqeucdistfuncp1(C,X);
            iter = iter + 1;
            if (iter > 250), fprintf('Maximum number of iterations occured\n'), break; end
        end     
        D = sqeucdistfuncp2(C,X,L);
        sumd = nansum(D);
        success = 1;
        
    case 'cit'
        
        X1 = X(~any(isnan(X),2),:);
        if isempty(start), start = X1(randi(size(X1,1)),:); end
        C = start;
        [~, L] = citydistfuncp1(C,X1);
        if (strcmp(initcrit,'kmeans++'))
            for i = size(C,1)+1:k
                D = cumsum(citydistfuncp2(C,X1,L));
                if D(end) == 0, C(i:k,:) = X1(ones(1,k-i+1),:); return; end
                C(i,:) = X1(find(rand < D/D(end),1),:);
                [~, L] = citydistfuncp1(C,X1);
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
        [~, L] = citydistfuncp1(C,X);
        L1 = 0;
        iter = 0;
        while any(L ~= L1)
            L1 = L;
            for i = 1:k, l = L==i; C(i,:) = medianfun(X(l,:)); end
            if any(any(isnan(C))), return; end
            [~, L] = citydistfuncp1(C,X);
            iter = iter + 1;
            if (iter > 250), fprintf('Maximum number of iterations occured\n'), break; end
        end
        D = citydistfuncp2(C,X,L);
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
        L = eucdistfuncp1(C,X);
        L1 = 0;
        iter = 0;
        while any(L ~= L1)
            L1 = L;
            for i = 1:k, l = L==i; C(i,:) = spatialmedianfun(X(l,:),C(i,:),100,1e-5); end
            if any(any(isnan(C))), return; end
            L = eucdistfuncp1(C,X);
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
% Compute nearest centroids for each observation using squared Euclidean 
% distance.
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

function D = sqeucdistfuncp2(C, X, L) 
% Description: 
% Compute squared Euclidean distances between observation and centroid 
% matrices.
% 
% Inputs:
%        C - Matrix of cluster centroids
%        X - Input data set
%        L - Cluster labels for each observation
%
% Output:
%        D - Distances to nearest centroids
%
D = bsxfun(@minus,X,C(L,:));
D = nansum(D.^2,2);

end

function [D, L] = citydistfuncp1(C, X) 
% Description: 
% Compute nearest centroids for each observation using City block distance.
%
% Inputs:
%        C - Matrix of cluster centroids
%        X - Input data set
%
% Outputs:
%        D - Distances to nearest centroids
%        L - Cluster labels for each observation
%
[nx, p] = size(X);
[nc, ~] = size(C);
D = zeros(nc,nx,'double');
for i = 1:nc
    dsq = zeros(nx,1,'double');
    for q = 1:p
        dsq1 = abs(X(:,q)-C(i,q));
        dsq1(isnan(dsq1)) = 0; 
        dsq = dsq + dsq1; 
    end
    D(i,:) = dsq;
end
[D, L] = min(D,[],1);
L = L(:);

end

function D = citydistfuncp2(C, X, L) 
% Description: 
% Compute City block distances between observation and centroid matrices.
%
% Inputs:
%         C - Matrix of cluster centroids
%         X - Input data set
%         L - Cluster labels for each observation
%
% Output:
%         D - Distances to nearest centroids
%
D = bsxfun(@minus,X,C(L,:));
D = nansum(abs(D),2);

end

function L = eucdistfuncp1(C, X) 
% Description: 
% Compute nearest centroids for each observation using Euclidean distance.
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

function D = eucdistfuncp2(C, X, L)
% Description: 
% Compute Euclidean distances between observation and centroid matrices.
%
% Inputs:
%         C - Matrix of cluster centroids
%         X - Input data set
%         L - Cluster labels for each observation 
%
% Output:
%         D - Distances to nearest centroids 
%
D = bsxfun(@minus,X,C(L,:));
D = sqrt(nansum(D.^2,2));

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

function C = medianfun(X)
% Description: 
% Compute median value of input data set. 
%
% Input:
%         X - Input data set 
%
% Output:
%         C - Median value of data set
%
Xsorted = sort(X,1);
count = sum(~isnan(X),1);
nn = floor(0.5*count);
n = size(X,2);
C = nan(1,n);
for j = 1:n
    if count(j) == 0
        C(1,j) = NaN;
    elseif mod(count(j),2) == 0
        C(1,j) = 0.5*(Xsorted(nn(j),j)+Xsorted(nn(j)+1,j));
    else
        C(1,j) = Xsorted(nn(j)+1,j);
    end
end

end

function C = spatialmedianfun(X, u, max_iter, tol)
% Description: 
% Compute spatial median value of input data set. 
%
% Inputs: 
%         X - Input data set
%         u - Previous value of cluster centroid
%  max_iter - Maximum number of iterations
%       tol - Tolerance value of convergence
%
% Output: 
%         C - Spatial Median value of data set
%
P = ~isnan(X);
iters = 0;
w = 1.5;
X(isnan(X)) = 0;
while iters < max_iter
    iters = iters + 1;
    D = P.*bsxfun(@minus,X,u);
    a = 1./sqrt(sum(D.^2,2)+sqrt(eps));
    a = bsxfun(@times,P,a);
    ax = sum(bsxfun(@times,a,X));
    v = (1./sum(a)).*ax;
    u1 = u + w*(v-u);
    if norm(u1-u,inf) < tol
        break;
    end
    u = u1;
end
C = u1;

end

