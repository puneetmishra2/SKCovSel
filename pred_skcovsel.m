function Yhat = pred_skcovsel(X,B)
% Unfold and add ones
X = [ones(size(X{1},1),1), cell2mat(cellfun(@unfold, X, 'UniformOutput',false))];

% Predict
Yhat = GMP(X,B,1);
end

%% Unfold array
function unfolded = unfold(X)
unfolded = reshape(X,size(X,1),[]);
end


%% Generalized matrix product, equivalent to https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html
function Z = GMP(X, Y, nactive)
% nactive - number of active dimensions of X and Y (shared dimensions)
%         - defaults to 1 if not specified
%         - if equal to 0 it results in "element-wise outer-product" 
%         - when nactive > 1, (A*B)*C ~= A*(B*C)
%         - when all nactive == 1 it seems like (A*B)*C == A*(B*C)
if nargin < 3
    nactive = 1;
end

% Dimensions
nx = size(X);
ny = size(Y);
xout = length(nx) - nactive; % Number of passive dimensions for X
if length(ny) == nactive
    ny = [ny,1];
end

Z = reshape( ...
    reshape(X,[prod(nx(1:xout)),prod(ny(1:nactive))]) * ...
    reshape(Y,[prod(ny(1:nactive)), prod(ny((nactive+1):end))]), ...
    [nx(1:xout), ny((nactive+1):end)]);
end

