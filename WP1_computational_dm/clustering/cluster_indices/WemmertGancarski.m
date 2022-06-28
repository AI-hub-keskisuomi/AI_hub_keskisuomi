function WG = WemmertGancarski(X, C, L, distance)
% Description: 
% Estimate Wemmert-Gancarski index with missing values. 
%
% Function call:
%        WG = WemmertGancarski(X, C, L, distance)
%
% Input:
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
%       WG - Value of Wemmert-Gancarski index
%
k = size(C,1);
if k == 1, WG = Inf; return; end
n = size(C,2);
Inter = 0;
C2 = C;
for i=1:k
    C = C2;
    I = find(L == i);
    C(i,:) = realmax/(10^6)*ones(1,n);
    OthClustdists = min(nandistfunc(X(I,:),C,distance),[],2);
    RM = nanmatrixdist(X(I,:),C2(i,:),distance)./OthClustdists;
    Inter = Inter + length(I) - nansum(RM);
end
WG = Inter;
% Inverse
WG = 1 / WG;

end

