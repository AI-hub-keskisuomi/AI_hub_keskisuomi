function CH = CalinskiHarabasz(X, C, L, distance)
% Description: 
% Estimate Calinski-Harabasz index with missing values.
%
% Function call:
%        CH = CalinskiHarabasz(X, C, L, distance)
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
%        CH - Value of Calinski-Harabasz index 
%
dist2 = distance;
if strcmp(distance,'euc'), dist2 = 'sqe'; end
if strcmp(distance,'cit'), dist2 = 'sqcit'; end
K = size(C,1);
if K == 1, CH = Inf; return; end
N = size(X,1);
Ni = zeros(K,1);
for i = 1:K, l = find(L==i); Ni(i) = length(l); end
Intra = nansum(nandistfuncp2(C,X,L,dist2));
Inter = nansum(Ni.*nanmatrixdist(C,nancentroid(X,distance),dist2));
CH = ((N-K)/(K-1))*(Inter/Intra);  
% Inverse
CH = 1 / CH;

end

