function PBMVal = PBM(X, C, L, distance)
% Description: 
% Estimate Pakhira-Bandyopadhyay-Maulik index with missing values.
%
% Function call:
%    PBMVal = PBM(X, C, L, distance)
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
%       PBMVal - Value of Pakhira-Bandyopadhyay-Maulik index 
%
k = size(C,1);
if k == 1, PBMVal = Inf; return; end
Intra = nansum(nanmatrixdist(X,C(L,:),distance));
Inter = max(nanpdistfunc(C,distance));
err = nansum(nanmatrixdist(X,nancentroid(X,distance),distance));
PBMVal = (Inter*err)/(k*Intra);
% Inverse
PBMVal = 1 / PBMVal;
 
end

