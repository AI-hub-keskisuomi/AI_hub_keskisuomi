function Ximp = knnimpute(Xm, varargin)
% Description: 
% Perform k-nearest neighbors imputation for the data set with 
% missing values. 
%
% Function call(s):
%         Ximp = knnimpute(Xm)
%         Ximp = knnimpute(Xm, K)
%         Ximp = knnimpute(Xm, K, distance)
%         Ximp = knnimpute(Xm, K, distance, distFunc)   
%
% Inputs:
%           Xm - Original data set with missing values
%            K - Number of nearest neighbors. Imputation is performed 
%                in complete manner if K = 1. Otherwise centroid values of 
%                k-nearest neighbors will be used. Default value: 5    
%     distance - Selected distance metric. Default value: 'euc' 
%                Alternatives: 
%                'sqe' - squared Euclidean distance
%                'euc' - Euclidean distance
%                'cit' - City block distance 
%     distFunc - Used distance estimation strategy. Default value: 'ads'  
%                Alternatives:
%                'ads' - Available data 
%                'pds' - Partial distance 
%                'exp' - Expected distance (ESD/EED)
%                
% Outputs:
%        Ximp - Imputed data set 
%
defaultK = 5;
defaultDistance = 'euc';
defaultDistFunc = 'ads';
%
p = inputParser;
addOptional(p, 'K', defaultK, @(x) isnumeric(x));
addOptional(p, 'distance', defaultDistance, ... 
                @(x) ismember(x,{'sqe','euc','cit'}));
addOptional(p, 'distFunc', defaultDistFunc, ... 
                @(x) ismember(x,{'ads','pds','exp'}));
parse(p,varargin{:});
K = p.Results.K;
distance = p.Results.distance;
distFunc = p.Results.distFunc;
%
% if strcmp(distFunc, 'ads')
%     addpath('../../toolbox/distance_functions/available');
%     addpath('../../toolbox/centroids/available');
% elseif strcmp(distFunc, 'pds')
%     addpath('../../toolbox/distance_functions/partial');
%     addpath('../../toolbox/centroids/partial');    
% elseif strcmp(distFunc, 'exp')
%     addpath('../../toolbox/distance_functions/expected');
%     addpath('../../toolbox/centroids/available');
% end
%
Xorg = Xm;
I = ~all(isnan(Xm),2);
Xm = Xm(I,:);
Ximp = Xm;
Im = find(any(isnan(Xm),2));
Ic = find(~any(isnan(Xm),2)); 
if strcmp(distFunc,'exp')
    [Xi, sx, ~] = ecmnmlefunc(Xm);
end
switch K
    % Nearest complete observation
    case 1
        for i = 1:length(Im)
            Xmi = Xm(Im(i),:);
            if strcmp(distFunc,'exp')
                D = nandistfunc(Xi(Im(i),:),Xm(Ic,:),distance,sx(Im(i),:),sx(Ic,:));
            else
                D = nandistfunc(Xmi,Xm(Ic,:),distance); 
            end
            [~, idx] = min(D);
            nearest = Xm(Ic(idx),:);
            ind = find(isnan(Xmi));
            Xmi(ind) = nearest(ind);
            Ximp(Im(i),:) = Xmi;
        end
    % Centroid of K nearest observations
    otherwise
        for i = 1:length(Im)
            Xmi = Xm(Im(i),:);
            if strcmp(distFunc,'exp')
                D = nandistfunc(Xi(Im(i),:),Xi,distance,sx(Im(i),:),sx);
            else
                D = nandistfunc(Xmi,Xm,distance);
            end
            [~, idx] = sort(D);
            cnt = 0;
            % In the case data vector consists missing values after 
            % imputation, K will be increased one by one until vector is 
            % fully complete. 
            while sum(isnan(Xmi)) > 0
                knearests = Xm(idx(2:K+1+cnt),:);
                centroid = nancentroid(knearests,distance);
                ind = find(isnan(Xmi));
                Xmi(ind) = centroid(ind);
                Ximp(Im(i),:) = Xmi;
                cnt = cnt + 1;
            end
        end
end
% Impute fully incomplete vectors
if (sum(I)<size(Xorg,1))
    Xcent = nancentroid(Xorg,distance);
    X0 = zeros(size(Xorg));
    cnt = 1;
    for i = 1:length(I)
        if I(i) == 1
            X0(i,:) = Ximp(cnt,:);
            cnt = cnt + 1;
        else
            X0(i,:) = Xcent;
        end
    end
    Ximp = X0;
end
%
% if strcmp(distFunc, 'ads')
%     rmpath('../../toolbox/distance_functions/available');
%     rmpath('../../toolbox/centroids/available');
% elseif strcmp(distFunc, 'pds')
%     rmpath('../../toolbox/distance_functions/partial');
%     rmpath('../../toolbox/centroids/partial');    
% elseif strcmp(distFunc, 'exp')
%     rmpath('../../toolbox/distance_functions/expected');
%     rmpath('../../toolbox/centroids/available');    
% end
end

