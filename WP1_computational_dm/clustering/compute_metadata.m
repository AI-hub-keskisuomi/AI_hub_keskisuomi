clear, clc, close all;
restoredefaultpath;
addpath('preprocess');
addpath('distance_functions');
addpath('kcentroids');
load('psykologiset_36');
load('terv_36');
load('paino_36');
load('relative_changes');
% Normalization
%
Xnorm = normalizedata(psykologiset_36, 'min-max', [-1,1]);
% Clustering
%
[L, C, sumd] = kcentroids(Xnorm, 7, 100, 'euc', 'kmeans++', []);
% Metadata
%
for i = 1:max(L)
    L1 = (L==i);
    sum(L1)
    mean(relative_changes(L1))
    median(relative_changes(L1))
%     mean(psykologiset_36(L1,:))
    median(psykologiset_36(L1,:))
%     figure;
%     histogram(terv_36(L1));
%     mean(paino_36(L1))
%     median(paino_36(L1))  
    fprintf('================\n');
end
% mean(psykologiset_36)
% median(psykologiset_36)
% fprintf('===========\n');
close all;
histogram(terv_36)
%
% Poislähteneet
% load('ID_0');
% load('ID_36');
% %mean(paino_0(~ismember(ID_0,ID_36)))
% mean(psykologiset_0(~ismember(ID_0,ID_36),:))
% median(psykologiset_0(~ismember(ID_0,ID_36),:))
% close all;
% histogram(terv_0(~ismember(ID_0,ID_36),:))
