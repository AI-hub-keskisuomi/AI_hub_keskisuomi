function D = nandistfuncp2(C, X, L, distance) 
% Description: 
% Estimate distances between observations with missing values and nearest 
% centroids. Uses available data strategy for treating missing values. 
%
% Function call:
%         D = nandistfuncp2(C, X, L, distance)
%
% Inputs:
%         C - Matrix of cluster centroids
%         X - Input data set
%         L - Cluster labels for each observation
%  distance - Selected distance metric 
%             Alternatives: 
%             'sqe' - squared Euclidean distance
%             'euc' - Euclidean distance 
%             'cit' - City block distance 
%             'sqcit' - squared City block distance 
%
% Output:
%         D -  Distances to nearest centroids 
%
switch distance
    case 'sqe'
        D = bsxfun(@minus,X,C(L,:));
        D = nansum(D.^2,2);
    case 'euc'
        D = bsxfun(@minus,X,C(L,:));
        D = nansum(D.^2,2);
        D = sqrt(D);        
    case 'cit'
        D = bsxfun(@minus,X,C(L,:));
        D = nansum(abs(D),2);
    case 'sqcit'
        D = bsxfun(@minus,X,C(L,:));
        D = nansum(abs(D).^2,2);        
end
        
end

