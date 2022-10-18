function Silh = Silhouette2(X, C, L, distance)
% Description: 
% Estimate Silhouette* index with missing values. Algorithm is speeded-up 
% version of original Silhouette index. More specify, distances are estimated 
% between observations and nearest centroids instead between observations 
% and another observations.
%
% Function call:
%      Silh = Silhouette2(X, C, L, distance)
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
%       Silh - Value of Silhouette* index   
%
clusts = unique(L);
num = length(clusts);
if num == 1, Silh = Inf; return; end
avgDWithin = Inf(num,1);
avgDBetween = Inf(num,num);
for i = 1:num
    members = (L == clusts(i));
    for j = 1:num
        if j==i
            avgDWithin(i) = nancentroid(nanmatrixdist(X(members,:), ...
                                C(j,:),distance),'sqe');
        else
            avgDBetween(i,j) = nancentroid(nanmatrixdist(X(members,:), ...
                                C(j,:),distance),'sqe');
        end  
    end  
end
minavgDBetween = min(avgDBetween, [], 2);
Silh = (minavgDBetween - avgDWithin) ./ minavgDBetween;
Silh = nancentroid(Silh,'sqe');
% Inverse
Silh = 1 / Silh;

end

