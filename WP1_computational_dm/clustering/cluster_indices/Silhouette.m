function Silh = Silhouette(X, ~, L, distance)
% Description: 
% Estimate Silhouette index with missing values. 
%
% Function call:
%      Silh = Silhouette(X, ~, L, distance)
%
% Inputs:
%         X - Input data set 
%         C - Matrix of cluster centroids 
%         L - Cluster labels for each observation
%  distance - Selected distance metric 
%             Alternatives: 
%             'euc' - Euclidean distance 
%             'sqe' - squared Euclidean distance
%             'cit' - City block distance 
%
% Output:
%       Silh - Value of Silhouette index   
%
cnames = unique(L);
k = length(cnames);
if k == 1, Silh = Inf; return; end
n = length(L);
mbrs = (repmat(1:k,n,1) == repmat(L,1,k));
avgDWithin = Inf(n,1);
avgDBetween = Inf(n,k);

for j = 1:n
    distj = nansum(nanmatrixdist(X,X(j,:),distance),2);
    for i = 1:k
        if i == L(j)
            mbrs1 = mbrs;
            mbrs1(j,i) = 0;
            avgDWithin(j) = nancentroid(distj(mbrs1(:,i)),'sqe');
        else
            avgDBetween(j,i) = nancentroid(distj(mbrs(:,i)),'sqe');
        end
    end
end
minavgDBetween = min(avgDBetween, [], 2);
Silh = (minavgDBetween - avgDWithin) ./ max(avgDWithin,minavgDBetween);
Silh = nancentroid(Silh,'sqe');
% Inverse
Silh = 1 / Silh;

end

