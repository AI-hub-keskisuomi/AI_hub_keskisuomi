clear, clc, close all;
% Correct IDs
%
% load('ID_changes');
% load('ID_36');
% load('terv_36');
% load('paino_36');
% terv_36 = terv_36(ismember(ID_36,ID_changes));
% paino_36 = paino_36(ismember(ID_36,ID_changes));
% ID_36 = ID_36(ismember(ID_36,ID_changes));
%
% Impute missing values
% load('psykologiset_changes');
% psykologiset_changes(isnan(psykologiset_changes)) = 0;
% load('psykologiset_0');
% for i = 1:size(psykologiset_0,2)
%     X1 = psykologiset_0(:,i);
%     X1(isnan(X1)) = median(X1,'omitnan');
%     psykologiset_0(:,i) = X1;
% end
%
% Compute array psykologiset_36
% load('ID_0');
% load('ID_36');
% load('psykologiset_0');
% load('psykologiset_changes');
% psykologiset_36 = psykologiset_0(ismember(ID_0,ID_36),:) + psykologiset_changes; 
%
% Scale terv_0 and terv_36 to the range [1, 3]
% load('terv_0');
% load('terv_36');
% terv_0(terv_0==2) = 1;
% terv_0(terv_0==3) = 2;
% terv_0(terv_0==4) = 3;
% terv_0(terv_0==5) = 3;
% terv_36(terv_36==2) = 1;
% terv_36(terv_36==3) = 2;
% terv_36(terv_36==4) = 3;
% terv_36(terv_36==5) = 3;

