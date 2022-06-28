function [Xm, I] = genmissdata(X, p)
% Description: 
% Generates missing values using the MCAR mechanism. Note: function 
% assumes that original data set is complete.  
%
% Function call: 
%     [Xm, I] = genmissdata(X, p)  
%
% Inputs:
%           X - Original data set without missing values
%           p - Portion of missing values 
%
% Outputs:
%           Xm - Data set with missing values 
%            I - Boolean values for data vectors which indicate that at 
%                least one available component exist
Xm = X;
samples = numel(X);
nans = randperm(samples);
nans = nans(1:uint32(p*samples));
Xm(nans) = NaN;
I = ~all(isnan(Xm),2);

end

