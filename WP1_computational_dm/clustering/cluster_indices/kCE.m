function kCEVal = kCE(X, C, L, distance)
% Description: 
% Estimate kCE-index with missing values. 
%
% Function call:
%    kCEVal = kCE(X, C, L, distance)
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
%       kCEVal - Value of kCE-index     
%
k = size(C,1);
switch distance
    case 'euc'
        kCEVal = k*nansum(nandistfuncp2(C,X,L,'sqe'),1);
    case 'sqe'
        kCEVal = k*nansum(nandistfuncp2(C,X,L,distance),1);
    case 'cit'
        kCEVal = k*nansum(nandistfuncp2(C,X,L,'sqcit'),1); 
end

