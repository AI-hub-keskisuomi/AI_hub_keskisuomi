function Ximp = ICknnimpute(Xm, varargin)
% Description: 
% Perform Incomplete-Case k-nearest neighbors imputation for the data set with 
% missing values. 
%
% Function call(s):
%         Ximp = ICknnimpute(Xm)
%         Ximp = ICknnimpute(Xm, K)
%         Ximp = ICknnimpute(Xm, K, distance)
%         Ximp = ICknnimpute(Xm, K, distance, distFunc)   
%
% Inputs:
%           Xm - Original data set with missing values
%            K - Number of nearest neighbors. Default value: 5    
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
%       Ximp - Imputed data set 
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
[N, n] = size(Xm);
Im = find(any(isnan(Xm),2));
Ximp = Xm;
if strcmp(distFunc,'exp')
    [Xi, sx, ~] = ecmnmlefunc(Xm);
end
for i = 1:N
    if ~ismember(i,Im)
        continue;
    end
    Xmi = Xm(i,:);
    miss = find(isnan(Xmi));
    notmiss = find(~isnan(Xmi));
    if strcmp(distFunc,'exp')
        D = nandistfunc(Xi(i,:),Xi,distance,sx(i,:),sx);
    else
        D = nandistfunc(Xmi,Xm,distance);
    end
    [~, idx] = sort(D);
    neighbors = zeros(length(miss),K);
    Kcounter = zeros(length(miss),1);
    for j = 2:length(idx)
        Xj = Xm(idx(j),:);
        miss2 = find(isnan(Xj));
        if sum(ismember(miss2,notmiss)) > 0
            continue;
        else
            for k = 1:length(miss)
                if ismember(miss(k),miss2) || Kcounter(k) == K
                    continue;
                else
                    Kcounter(k) = Kcounter(k) + 1;
                    neighbors(k, Kcounter(k)) = idx(j);
                end
            end
        end
        if isempty(find(neighbors==0,1))
            break;
        end
    end
    for j = 1:length(miss)
        if Kcounter(j) > 0
            neighbors1 = neighbors(j,1:Kcounter(j));
            Xmi(miss(j)) = sum(Xm(neighbors1,miss(j)))/Kcounter(j);
        end
    end
    Ximp(i,:) = Xmi;
end
if ~isempty(isnan(Ximp))
    for i = 1:n
        X1 = Ximp(:,i);
        X1(isnan(X1)) = nancentroid(Xm(:,i),distance);
        Ximp(:,i) = X1;
    end
end
%
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

end

