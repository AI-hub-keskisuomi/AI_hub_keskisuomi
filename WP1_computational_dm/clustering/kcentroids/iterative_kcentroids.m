function [centers, labels] = iterative_kcentroids(X, varargin)
% Description: 
% Perform kcentroids clustering iteratively from 2 to maxClusters based on 
% selected clustering algorithm.
%
% Function calls:
% [centers, labels] = iterative_kcentroids(X)
% [centers, labels] = iterative_kcentroids(X, maxClusters)
% [centers, labels] = iterative_kcentroids(X, maxClusters, clustMethod)
% [centers, labels] = iterative_kcentroids(X, maxClusters, clustMethod, distance)
% [centers, labels] = iterative_kcentroids(X, maxClusters, clustMethod, distance, replicates)
% [centers, labels] = iterative_kcentroids(X, maxClusters, clustMethod, distance, replicates, useprevCent)
% [centers, labels] = iterative_kcentroids(X, maxClusters, clustMethod, distance, replicates, useprevCent, initcrit)
% [centers, labels] = iterative_kcentroids(X, maxClusters, clustMethod, distance, replicates, useprevCent, initcrit, pipelined)
% [centers, labels] = iterative_kcentroids(X, maxClusters, clustMethod, distance, replicates, useprevCent, initcrit, pipelined, showProgression)
%
% Inputs:
%                  X - Input data set
%        maxClusters - Maximum number of clusters. Default value: 20 
%        clustMethod - Selected clustering algorithm. 
%                      Default value: @kcentroids_expected 
%                      Alternatives: 
%                      @kcentroids - Available data based clustering 
%                      @kcentroids_partial - Partial distances based clustering 
%                      @kcentroids_expected - Expected distances based clustering
%           distance - Selected distance metric. Default value: 'euc'
%                      Alternatives: 
%                      'sqe' - squared Euclidean distance
%                      'euc' - Euclidean distance
%                      'cit' - City block distance 
%                      Note: @kcentroids_expected does not support 'cit' option 
%         replicates - Number of repetitions. Default value: 100
%        useprevCent - Use previous centers in initialization. 
%                      Default value: true                   
%           initcrit - Initialization criterion. Default value: 'kmeans++' 
%                      Alternatives: 
%                      'random' - Random selection of initial points
%                      'kmeans++' - Kmeans++ based selection of initial points
%          pipelined - Pipeline expected distances based clustering results. 
%                      Default value: true 
%                      Note: @kcentroids_expected only uses pipeline option  
%    showProgression - Boolean value which indicate if progression of clustering   
%                      is presented. Default value: true
% Outputs:
%        centers - Obtained cluster centers
%         labels - Obtained cluster labels
%
defaultMaxClusters = 20;
defaultClustMethod = @kcentroids_expected;
defaultDistance = 'euc';
defaultReplicates = 100;
defaultUseprevCent = true;
defaultInitcrit = 'kmeans++';
defaultPipelined = true;
defaultshowProgression = true;
%
p = inputParser;
addOptional(p, 'maxClusters', defaultMaxClusters, @(x) isnumeric(x));
addOptional(p, 'clustMethod', defaultClustMethod);
addOptional(p, 'distance', defaultDistance, @(x) ismember(x,{'sqe','euc','cit'}));
addOptional(p, 'replicates', defaultReplicates, @(x) isnumeric(x));
addOptional(p, 'useprevCent', defaultUseprevCent, @islogical);
addOptional(p, 'initcrit', defaultInitcrit, @(x) ismember(x,{'random','kmeans++'}));
addOptional(p, 'pipelined', defaultPipelined, @islogical);
addOptional(p, 'showProgression', defaultshowProgression, @islogical);
parse(p, varargin{:});
%
maxClusters = p.Results.maxClusters;
clustMethod = p.Results.clustMethod;
distance = p.Results.distance;
replicates = p.Results.replicates;
useprevCent = p.Results.useprevCent;
initcrit = p.Results.initcrit;
pipelined = p.Results.pipelined;
showProgression = p.Results.showProgression;
%
C = [];
centers = cell(maxClusters-1,1);
labels = cell(maxClusters-1,1);
%
fprintf('Performing clustering...\n');
for k = 2:maxClusters
    if showProgression
        if k~= maxClusters, fprintf('k: %d, ',k); else, fprintf('k: %d\n',k); end 
        if mod(k,10) == 0 && k~= maxClusters, fprintf('\n'); end
    end
    if ~useprevCent, C = []; end
    [L, C, ~] = clustMethod(X,k,replicates,distance,initcrit,C);
    centers{k-1} = C;
    labels{k-1} = L;
end
%
% Pipelined clustering:
if (strcmp(char(clustMethod),'kcentroids_expected') && pipelined)
    for k = 2:maxClusters
        [labels{k-1}, centers{k-1}, ~] = ...
            kcentroids(X,k,replicates,distance,initcrit,centers{k-1});
    end
end
fprintf('Done! \n');

end

