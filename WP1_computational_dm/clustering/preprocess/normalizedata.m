function X = normalizedata(X, method, varargin)
% Description: 
% Rescale columns and rows of input data set based on min-max or z-score 
% normalization.  
%
% Function calls:
%         X = normalizedata(X, method)
%         X = normalizedata(X, method, range)
%
% Inputs:
%         X - Input data set
%    method - Pre-processing method 
%             Alternatives: 
%             'min-max' - Min-max normalization 
%             'zscore' - Z-score normalization
%     range - Range of min-max scaling. Default value: [-1, 1]  
%
% Output:
%         X - Pre-procesed data set
%
defaultRange = [-1, 1];
p = inputParser;
addOptional(p, 'range', defaultRange, @(x) isnumeric(x));
parse(p,varargin{:});
range = p.Results.range;
%
switch method
    case 'min-max'
        a = range(1);
        b = range(2);
        min_x = min(X);
        max_x = max(X);
        cofs = (b - a) ./ (max_x - min_x);
        if (b - a) <= sqrt(eps), error('Range must be non-negative!'); end
        X = bsxfun(@minus,X,min_x);
        X = a + bsxfun(@times,X,cofs);
    case 'zscore'
        X = bsxfun(@minus,X,mean(X,'omitnan'));
        X = bsxfun(@rdivide,X,std(X,'omitnan'));
    otherwise
        fprintf('Nothing to be done...\n')
end

end

