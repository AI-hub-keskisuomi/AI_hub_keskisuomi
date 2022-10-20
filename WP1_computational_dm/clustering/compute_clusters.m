clear, clc, close all;
restoredefaultpath;
load('psykologiset_36');
addpath('preprocess');
addpath('distance_functions');
addpath('kcentroids');
addpath('cluster_indices')
% Normalization
%
Xnorm = normalizedata(psykologiset_36, 'min-max', [-1,1]);
%Xnorm = normalizedata(psykologiset_0, 'zscore');
% Index values
%
[centers, labels] = iterative_kcentroids(Xnorm, 10, @kcentroids, 'euc', 300, true, 'kmeans++', false, true);
indices = [{@CalinskiHarabasz, 'sqe'}; {@DaviesBouldin, 'euc'}; {@kCE, 'sqe'}; {@PBM, 'euc'}; ...
    {@RayTuri, 'sqe'}; {@Silhouette, 'euc'}; {@WB, 'sqe'}; {@WemmertGancarski, 'euc'}];
indices_values = cluster_validation(Xnorm, centers, labels, 'euc', indices, 'ads');
plot_indices(indices, indices_values);
