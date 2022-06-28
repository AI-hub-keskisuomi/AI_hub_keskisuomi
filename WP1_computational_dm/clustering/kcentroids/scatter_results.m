function scatter_results(X, C, L)
% Description: 
% Visualize clustering result for two-dimensional data.
%
% Function call:
%          scatter_results(X, C, L)
%
% Input:
%          X - Input data set 
%          C - Cluster centroids
%          L - Cluster labels for each observation 
%
kNumber = size(C,1); 
figure;
hold on;
box on;
axis square;
colors = lines(kNumber);
title('Clustering result');
for i = 1:kNumber
    scatter(X(L==i,1),X(L==i,2),'CData',colors(i,:));
    scatter(C(i,1),C(i,2),50,'ko','filled');
end

end

