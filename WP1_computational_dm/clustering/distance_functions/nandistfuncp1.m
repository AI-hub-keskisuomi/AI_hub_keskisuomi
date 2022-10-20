function L = nandistfuncp1(C, X, distance)
% Description: 
% Estimate nearest centroids for each observation with missing values. 
% Uses available data strategy for treating missing values.
%
% Function call:
%         L = nandistfuncp1(C, X, distance)
%
% Inputs:
%         C - Matrix of cluster centroids
%         X - Input data set
%  distance - Selected distance metric 
%             Alternatives: 
%             'sqe' - squared Euclidean distance
%             'euc' - Euclidean distance 
%             'cit' - City block distance 
%
% Output:
%         L - Cluster labels for each observation
%
switch distance
    case 'sqe'
        I = sparse(isnan(X));
        X(I) = 0;
        [~, L] = min((C.^2)*~I'-2*C*X',[],1);
        L = L(:);
    case 'euc'
        I = sparse(isnan(C));
        I2 = sparse(isnan(X));
        X(I2) = 0;
        D = ((X.^2)*~I'-2*X*C'+~I2*(C.^2)');
        D(D<eps) = 0;
        [~, L] = min(sqrt(D),[],2);
        L = L(:);
    case 'cit'
        [nx, p] = size(X);
        [nc, ~] = size(C);
        D = zeros(nc,nx,'double');
        for i = 1:nc
            dsq = zeros(nx,1,'double');
            for q = 1:p
                dsq1 = abs(X(:,q)-C(i,q));
                dsq1(isnan(dsq1)) = 0;
                dsq = dsq + dsq1;
            end
            D(i,:) = dsq;
        end
        [~, L] = min(D,[],1);
        L = L(:);
end

end

