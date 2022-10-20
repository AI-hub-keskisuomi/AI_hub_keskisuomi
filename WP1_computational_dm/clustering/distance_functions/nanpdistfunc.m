function D = nanpdistfunc(X, distance)
% Description: 
% Estimate pairwise distances between all pairs of observations with 
% missing values in input data set. Uses available data strategy for 
% treating missing values. 
%
% Function call:
%           D = nanpdistfunc(X, distance)
%
% Inputs:
%         X - Input data set 
%  distance - Selected distance metric 
%             Alternatives: 
%             'sqe' - squared Euclidean distance
%             'cit' - City block distance 
%             'euc' - Euclidean distance 
%
% Output:
%           D - Distances between observations
%
switch distance
    case 'sqe'
        n = size(X,1);
        D = nandistfunc(X,X,'sqe');
        B = tril(ones(n,n),-1);
        D = D(B==1);
    case 'cit'
        n = size(X,1);
        D = nandistfunc(X,X,'cit');
        B = tril(ones(n,n),-1);
        D = D(B==1);
    case 'euc'
        n = size(X,1);
        D = nandistfunc(X,X,'sqe');
        B = tril(ones(n,n),-1);
        D = sqrt(D(B==1));        
end

end

