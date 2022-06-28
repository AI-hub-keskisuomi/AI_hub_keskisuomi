function RT = RayTuri(X, C, L, distance)
% Description: 
% Estimate Ray-Turi index with missing values. 
%
% Function call:
%        RT = RayTuri(X, C, L, distance)
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
%       RT - Value of Ray-Turi index    
%
K = size(C,1);
if K == 1, RT = Inf; return; end
Intra = nansum(nandistfuncp2(C,X,L,distance));
Inter = min(nanpdistfunc(C,distance));
RT = Intra / Inter;

end

