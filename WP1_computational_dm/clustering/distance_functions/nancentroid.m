function C = nancentroid(X, distance)
% Description: 
% Estimate centroid value of input data set. Algorithm uses available data 
% strategy for treating missing values.   
% 
% Function call: 
%         C = nancentroid(X, distance)
%
% Inputs: 
%         X - Input data set 
%  distance - Selected distance metric  
%             Alternatives:
%             'euc' - Euclidean distance
%             'sqe' - squared Euclidean distance
%             'cit' - City block distance
%
% Output: 
%         C - Centroid value of data set 
%
switch distance
    case 'euc'
        % Spatial median value of data
        C = nanspatialmedianfunc(X);
    case 'sqe'
        nan = isnan(X);
        X(nan) = 0;
        % Mean value of data 
        C(1,:) = sum(X,1) ./ sum(~nan,1);  
    case 'cit'
        Xsorted = sort(X,1);
        count = sum(~isnan(X),1);
        nn = floor(0.5*count);
        n = size(X,2);
        C = NaN(1,n);
        % Median value of data 
        for j = 1:n
            if count(j) == 0
                C(1,j) = NaN;
            elseif mod(count(j),2) == 0
                C(1,j) = 0.5*(Xsorted(nn(j),j)+Xsorted(nn(j)+1,j));
            else
                C(1,j) = Xsorted(nn(j)+1,j);
            end
        end
end

end

