function D = distancecalc(Xm, varargin)
% Description:
% Calculate pairwise distances between all observations in selected 
% data set. Function assumes that data contains missing values. Available data, 
% partial distance, expected squared euclidean, and expected euclidean strategies 
% are available algorithms. 
%
% Inputs:
%             dataset - Input data set 
%               myfun - Used distance computation function. 
%                       Default value: 'ad'
%                       Alternatives: 
%                       'ad' - Available data 
%                       'pds' - Partial distance 
%                       'esd' - Expected squared Euclidean
%                       'eed' - Expected Euclidean
%
% Output:
%                   D - Distance vector of calculated pairwise distances
%                   (number of pairs: (N-1) + (N-2) + ... (N-(N-1)))
%
defaultMyfun = 'ad';
%
p = inputParser;
addOptional(p, 'myfun', defaultMyfun, ... 
                @(x) ismember(x,{'ad','pds','esd','eed'}));
parse(p,varargin{:});
myfun = p.Results.myfun;
if strcmp(myfun,'ad')
    myfun = @AD;
elseif strcmp(myfun,'pds')
    myfun = @PDS;
elseif strcmp(myfun,'esd')
    myfun = @ESD;
elseif strcmp(myfun,'eed')
    myfun = @EED;
end
D = myfun(Xm);
D(isnan(D)) = mean(D,'omitnan');

end
