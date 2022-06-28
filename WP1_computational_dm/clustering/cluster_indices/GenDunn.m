function GDunn = GenDunn(X, C, L, distance)
% Description: 
% Estimate Generalized Dunn index with missing values.
%
% Function call:
%     GDunn = GenDunn(X, C, L, distance)
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
%       GDunn - Value of Generalized Dunn index  
%
clusts = unique(L);
num = length(clusts);
if num == 1, GDunn = Inf; return; end
Intra = zeros(num,1);
for i = 1:num
    members = (L == clusts(i));
    Intra(i) = nancentroid(nanmatrixdist(X(members,:),C(i,:), ...
                    distance),'sqe');
end
Intra = max(Intra);
Inter = min(nanpdistfunc(C,distance));
GDunn = Inter / Intra;
% Inverse
GDunn = 1 / GDunn;
              
end

