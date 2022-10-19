function DB = DaviesBouldin(X, C, L, distance)
% Description: 
% Estimate Davies-Bouldin index with missing values.
%
% Funcion call:
%        DB = DaviesBouldin(X, C, L, distance)
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
%       DB - Value of Davies-Bouldin index   
%
clusts = unique(L);
num = length(clusts);
if num == 1, DB = Inf; return; end
aveWithinD = zeros(num,1);
for i = 1:num
    members = (L == clusts(i));
    aveWithinD(i) = nancentroid(nanmatrixdist(X(members,:),C(i,:), ...
                       distance),'sqe');
end
interD = nanpdistfunc(C,distance);
R = zeros(num);
cnt = 1;
for i = 1:num
    for j = i+1:num
        R(i,j) = (aveWithinD(i)+aveWithinD(j)) / interD(cnt);
        cnt = cnt + 1;
    end
end
R = R + R';
RI = max(R,[],1);
DB = nancentroid(RI(:),'sqe');
       
end

