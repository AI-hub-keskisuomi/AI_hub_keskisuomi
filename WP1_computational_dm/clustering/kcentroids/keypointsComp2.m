function kp_idx = keypointsComp2(X, dist, clusterNum, keyPointsNum)
% Description: 
% Key point selection algorithm (v. 2.0). The algorithm selects initial key 
% points and removes points one-by-one until the final number of key points 
% is reached. 
%
% Function call:
% kp_idx = keypointsComp(X, dist, clusterNum, keyPointsNum)
%
% Inputs:
%            X - Input data set (NOTE: must be complete)
%         dist - Selected distance metric. Default value: 'euc' 
%                Alternatives: 
%                'sqe' - squared Euclidean distance
%                'euc' - Euclidean distance
%   clusterNum - The final number of clusters
% keyPointsNum - Initial number of key points
%
% Output:
%       kp_idx - Key point indexes
%
%Density
density = zeros(size(X,1),1);
for i = 1:size(X,1)
    Xi = X(i,:);
    D = pdist2(Xi,X,dist);
    [~, idx] = sort(D);
    knearests = X(idx(2:5),:);
    for j = 1:size(knearests,1)
        density(i) = density(i) + pdist2(Xi,knearests(j,:),dist);
    end
    density(i) = 1/density(i);
end
%
%Key points
sigma = inf(size(X,1),1);
kp = inf(size(X,1),1);
for i = 1:size(X,1)
    Xi = X(i,:);
    D = pdist2(Xi,X,dist);
    [~, idx] = sort(D);
    for j = 2:length(idx)
        if density(i) < density(idx(j)) 
            sigma(i) = pdist2(Xi,X(idx(j),:),dist);
            break;
        end
    end
    kp(i) = density(i)*sigma(i);
end
[~, kp_idx] = sort(kp,'descend');
kp_idx = kp_idx(1:keyPointsNum);
%
%Repeat selection step for key points
Xorg = X;
X = Xorg(kp_idx,:);
densOrg = density;
density = densOrg(kp_idx);
while size(X,1)>clusterNum
    %
    %Key points
    sigma = inf(size(X,1),1);
    kp = inf(size(X,1),1);
    for i = 1:size(X,1)
        Xi = X(i,:);
        D = pdist2(Xi,X,dist);
        [~, idx] = sort(D);
        for j = 2:length(idx)
            if density(i) < density(idx(j))
                sigma(i) = pdist2(Xi,X(idx(j),:),dist);
                break;
            end
        end
        kp(i) = density(i)*(sigma(i));
    end
    [~, new_idx] = sort(kp);
    I = find(kp_idx==kp_idx(new_idx(1)));
    kp_idx(I) = [];
    X = Xorg(kp_idx,:);
    density = densOrg(kp_idx);
    
end
 kp_idx = kp_idx(:);

 