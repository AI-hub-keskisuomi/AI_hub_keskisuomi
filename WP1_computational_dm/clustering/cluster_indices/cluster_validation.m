function indices_values = cluster_validation(X, centers, labels, varargin)
% Description: 
% Computes cluster validation indices values with missing values.
%
% Function calls:
% indices_values = cluster_validation(X, centers, labels, distance)
% indices_values = cluster_validation(X, centers, labels, distance, clusterInd)
% indices_values = cluster_validation(X, centers, labels, distance, clusterInd, distFunc)
%
% Inputs:
%              X - Input data set
%        centers - Cluster centers obtained by iterative_kcentroids function
%         labels - Cluster labels obtained by iterative_kcentroids function
%       distance - Selected distance metric. Default value: 'euc' 
%                  Alternatives:
%                  'sqe' - squared Euclidean distance
%                  'euc' - Euclidean distance
%                  'cit' - City block distance
%     clusterInd - Selected cluster validation indices
%                  Alternatives (multiple indices can be selected): 
%                  @CalinskiHarabasz - Calinski-Harabasz (Selected by default)
%                  @DaviesBouldin - Davies-Bouldin (Selected by default)
%                  @DaviesBouldin2 - Davies-Bouldin* (Selected by default)
%                  @GenDunn - Generalized Dunn (Selected by default)
%                  @kCE - kCE-index (Selected by default)
%                  @PBM - Pakhira-Bandyopadhyay-Maulik (Selected by default)
%                  @RayTuri - Ray-Turi (Selected by default)
%                  @Silhouette - Silhouette (Selected by default)
%                  @Silhouette2 - Silhouette*
%                  @WB - WB-index (Selected by default)
%                  @WemmertGancarski - Wemmert-Gancarski (Selected by default) 
%                  %
%                  Note: Used distance metric(s) for the selected indices can 
%                  be optionally separately specified. 
%                  Use then the following format typing: 
%                  [{@index1, distance1}; {@index2, distance2}; ...], e.g.,
%                  [{@CalinskiHarabasz, 'sqe'}; {@DaviesBouldin, 'euc'}]
%                  In general case, if specific distances were not given, 
%                  the 'distance' parameter will be used (see definition above). 
%
%       distFunc - Used distance estimation strategy for indices. 
%				   Default value: 'ads'
%                  Alternatives:
%                  'ads' - Available data 
%                  'pds' - Partial distance 
%                  'exp' - Expected distance (ESD/EED)
%
% Output:
%
% indices_values - Values of cluster validation indices.
%
defaultDistance = 'euc';
defaultClusterInd = {@CalinskiHarabasz, @DaviesBouldin, @GenDunn, ... 
    @kCE, @PBM, @RayTuri, @Silhouette, @WB, @WemmertGancarski};
defaultDistFunc = 'ads';
%
p = inputParser;
addOptional(p, 'distance', defaultDistance, ... 
                @(x) ismember(x,{'sqe','euc','cit'}));
addOptional(p, 'clusterInd', defaultClusterInd);
addOptional(p, 'distFunc', defaultDistFunc, ...
                @(x) ismember(x,{'ads','pds','exp'}));
%
parse(p,varargin{:});
distance = p.Results.distance;
clusterInd = p.Results.clusterInd;
distFunc = p.Results.distFunc;
%
indices_values = zeros(length(clusterInd), size(centers,1)+1);
I = ~all(isnan(X),2);
X = X(I,:);

fprintf('Performing cluster validation...\n');
for k = 2:size(centers,1)+1
    C = centers{k-1};
    L = labels{k-1};
    L = L(L~=0);
    L = L(:);
    if (strcmp(distFunc,'ads') || strcmp(distFunc,'pds'))
        for r = 1:size(clusterInd,1)
            clusterIndex = clusterInd{r,1};
            indices_values(r,1) = Inf(1,1);
            if size(clusterInd,2) > 1
                indexDist = clusterInd{r,2};
                indices_values(r,k) = clusterIndex(X,C,L,indexDist);
            else
                indices_values(r,k) = clusterIndex(X,C,L,distance);
            end
        end
    elseif (strcmp(distFunc,'exp'))
        Xi = zeros(size(X));
        sx = zeros(size(X));
        for i = 1:k
            [Xi(L==i,:), sx(L==i,:), success] = ecmnmlefunc(X(L==i,:));
            if ~success
                Xi(L==i,:) = X(L==i,:);
                sx(L==i,:) = zeros(size(sx(L==i,:)));
                for j = 1:size(Xi,2)
                    X1 = Xi(L==i,j);
                    X1(isnan(X1)) = median(X1,'omitnan');
                    Xi(L==i,j) = X1;
                end
            end
        end 
        for r = 1:size(clusterInd,1)
            clusterIndex = clusterInd{r,1};
            indices_values(r,1) = Inf(1,1);
            if size(clusterInd,2) > 1
                indexDist = clusterInd{r,2};
                indices_values(r,k) = clusterIndex(X,Xi,sx,C,L,indexDist);
            else
                indices_values(r,k) = clusterIndex(X,Xi,sx,C,L,distance);
            end
        end
    end
end
fprintf('Done! \n');

end

