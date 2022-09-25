function [B, T, W, Q, R, Wa, Pt, blockOrder, modeSel, inds, crbvar] = skcovsel(ncomp, X, Y, modeSel, blockOrder)
% -----------------------------------------------------
% ------------------ KHL+PM 2022 ----------------------
% -----------------------------------------------------
% --------------- Swiss Knife CovSel ------------------
% -----------------------------------------------------
% Input arguments
% ncomp   - number of components
% X       - data vector, matrix or tensor
% Y       - response vector or matrix
% Yadd    - (optional) addition response/meta information
% modeSel - (optional) modes for variable selection, vector of length ncomp
%           0 : (default) automatic selection
%           1 : single variable combination, 
%          >1 : single slice
% blockOrder - vector of fixed block order (0 for free choice)
if nargin < 4, modeSel = zeros(ncomp,1); end
if nargin < 5, blockOrder = zeros(ncomp,1); end
crbvar = zeros(ncomp,1);
% Initialize
nx = cellfun(@size, X, 'UniformOutput', false); ndim = cellfun(@length,nx);
N = nx{1}(1); ny = size(Y); nresp = ny(2); nblock = length(X);
W = cell(1,nblock); Wa_candidates = W; Pt_candidates = W; Pt = W; Wa = W'; i_candidates = W;
for b=1:nblock
    W{b} = cell(1,ndim(b)-1);
    for i=1:(ndim(b)-1)
        W{b}{i} = zeros(nx{b}(i+1), ncomp);
    end
    if ndim(b)>2, ddim = nx{b}(2:end); else, ddim = [nx{b}(2),1]; end
    Pt{b} = zeros([ncomp,ddim]); 
    Wa{b} = zeros([ddim,ncomp]);    
end
T = zeros(N, ncomp); Q = zeros(nresp, ncomp); inds = cell(1,ncomp);
t_candidates = zeros(N, nblock);

% Centre the matrices:
Xm  = mean(cell2mat(cellfun(@unfold, X, 'UniformOutput', false)));
Ym  = mean(Y,1);
X   = cellfun(@center, X, 'UniformOutput', false);
Y   = Y - Ym;
Xuf = cellfun(@unfold, X, 'UniformOutput', false);
vartot(1)= sum(sum(Y.*Y));

% Component loop
for a = 1:ncomp
    aBlocks = 1:nblock; if blockOrder(a)>0, aBlocks = blockOrder(a); end
    r_candidates = zeros(1,nblock);

    % Loop over block candidates or go directly to chosen block
    for b = aBlocks
        covar = sum(GMP(GT(X{b}, ndim(b)-1), Y, 1).^2, ndim(b));
        t_mode = zeros(N,ndim(b)); r_mode = zeros(1,ndim(b)); 
        w_mode = cell(nblock,1); i_mode = cell(1,ndim(b));

        % Single variable
        if modeSel(a) <= 1
            [~, argmax] = max(covar(:));
            t = Xuf{b}(:, argmax); 
            if a>1, t = t-T(:,1:a-1)*(T(:,1:a-1)'*t); end
            t = t./norm(t); t_mode(:,1) = t;
            [r_mode(1),~] = ccXY(t,Y, []);
            if ndim(b)>2, ddim = nx{b}(2:end); else, ddim = [nx{b}(2),1]; end
            w_mode{1} = zeros(ddim); w_mode{1}(argmax) = 1;
            i_mode{1} = ind2subVec(ddim, argmax);
        end

        % Slice
        if ndim(b) > 2 && (modeSel(a) == 0 || modeSel(a) > 1)
            wloads = cell(ndim(b),1);
            if ndim(b) == 3
                [wloads{1},~,wloads{2}] = svds(covar, 1);
            else
                wloads = parafac(covar,1,[0 2 0 0 NaN]');
                wloads{1} = wloads{1}./norm(wloads{1});
            end
            bBlock = 2:ndim(b); if modeSel(a) > 1, bBlock = modeSel(a); end

            % Loop over modes for slicing
            for m = bBlock
                [~,indm] = max(abs(wloads{m-1})); i_mode{m} = indm;
                wloadsm = wloads; % Exchange one vector with selection vector
                wloadsm{m-1} = zeros(size(wloads{m-1})); wloadsm{m-1}(indm) = 1;
                w_mode{m} = GOuter(wloadsm(1:ndim(b)-1));
                t = GMP(X{b},w_mode{m},ndim(b)-1); 
                if a>1, t = t-T(:,1:a-1)*(T(:,1:a-1)'*t); end
                t = t./norm(t); t_mode(:,m) = t;
                [r_mode(m),~] = ccXY(t,Y, []);
            end
        end

        % Winning mode
        [~,win_mode] = max(r_mode);
        t_candidates(:,b) = t_mode(:,win_mode);
        Wa_candidates{b}  = w_mode{win_mode};
        Pt_candidates{b}  = GMP(GT(X{b}, ndim(b)-1), t_mode(:,win_mode), 1);
        r_candidates(b)   = r_mode(win_mode);
        modeSel(a) = win_mode;
        i_candidates{b}   = i_mode{win_mode};
    end

    % Winning block
    [~,win_block] = max(r_candidates);
    blockOrder(a) = win_block;
    inds{a}       = i_candidates{win_block};

    % Accumulate
    T(:,a) = t_candidates(:,win_block);
    Pt{win_block} = insert(Pt{win_block},GMP(GT(X{win_block}, ndim(win_block)-1), T(:,a), 1),a,ndim(win_block));
    Wa{win_block} = insert(Wa{win_block},Wa_candidates{win_block},a,ndim(win_block),false);
    Q(:,a) = Y'*T(:,a);

    % Deflate
    Y = Y - T(:,a)*Q(:,a)';
    crbvar(a,1) = (vartot(1) - sum(sum(Y.*Y)) ) / vartot(1);
end
if ncomp == 1
    Ptu = cell2mat(cellfun(@vect, Pt, 'UniformOutput', false));
    Wau = cell2mat(cellfun(@vec, Wa, 'UniformOutput', false));
else
    Ptu = cell2mat(cellfun(@unfold, Pt, 'UniformOutput', false));
    Wau = cell2mat(cellfun(@unfoldt, Wa, 'UniformOutput', false));
end
R   = Wau/triu(Ptu*Wau);
B   = cumsum(R.*shiftdim(Q',-1), 2);
B   = cat(1,shiftdim(Ym,-1)-GMP(Xm,B,1), B);
end


%% Canonical correlations
function [r,A] = ccXY(X,Y,wt)
% Computes the coefficients in canonical variates between collumns of X and Y
[n,p1] = size(X); p2 = size(Y,2);

% Weighting of observations with regards to wt (asumes weighted centering already performed)
if ~isempty(wt)
    X = rscale(X,wt);
    Y = rscale(Y,wt);
end

% Factoring of data by QR decomposition and ellimination of internal linear
% dependencies in X and Y
[Q1,T11,perm1] = qr(X,0);       [Q2,T22,~] = qr(Y,0);
rankX          = sum(abs(diag(T11)) > eps(abs(T11(1)))*max(n,p1));
rankY          = sum(abs(diag(T22)) > eps(abs(T22(1)))*max(n,p2));
if rankX < p1
    Q1 = Q1(:,1:rankX); T11 = T11(1:rankX,1:rankX);
end
if rankY < p2
    Q2 = Q2(:,1:rankY);
end

% Economical computation of canonical coefficients and canonical correlations
d = min(rankX,rankY);
if nargout == 1
    D    = svd(Q1' * Q2,0);
    r    = min(max(D(1:d), 0), 1); % Canonical correlations
else
    [L,D]    = svd(Q1' * Q2,0);
    A        = T11 \ L(:,1:d) * sqrt(n-1);
    % Transform back coefficients to full size and correct order
    A(perm1,:) = [A; zeros(p1-rankX,d)];
    r = min(max(diag(D(1:d)), 0), 1); % Canonical correlations
end
end

%% Insert into array
function Z = insert(Z,X,i,ndim,first)
%ndim = length(size(Z));
if nargin < 5
    first = true;
end
if first
    switch ndim
        case 2
            Z(i,:) = X;
        case 3
            Z(i,:,:) = X;
        case 4
            Z(i,:,:,:) = X;
        case 5
            Z(i,:,:,:,:) = X;
        case 6
            Z(i,:,:,:,:,:) = X;
        case 7
            Z(i,:,:,:,:,:,:) = X;
        case 8
            Z(i,:,:,:,:,:,:,:) = X;
        case 9
            Z(i,:,:,:,:,:,:,:,:) = X;
        case 10
            Z(i,:,:,:,:,:,:,:,:,:) = X;
        otherwise
            error('Wrong size of Z')
    end
else % last
    switch ndim
        case 2
            Z(:,i) = X;
        case 3
            Z(:,:,i) = X;
        case 4
            Z(:,:,:,i) = X;
        case 5
            Z(:,:,:,:,i) = X;
        case 6
            Z(:,:,:,:,:,i) = X;
        case 7
            Z(:,:,:,:,:,:,i) = X;
        case 8
            Z(:,:,:,:,:,:,:,i) = X;
        case 9
            Z(:,:,:,:,:,:,:,:,i) = X;
        case 10
            Z(:,:,:,:,:,:,:,:,:,i) = X;
        otherwise
            error('Wrong size of Z')
    end
end
end

%% Generalised transpose, 
% Switch active and passive dimensions of an array
function [X, nleft] = GT(X, nright)
% nright - number of right facing dimensions, e.g. active dimensions.
%        - defaults to 1 if not specified
if nargin == 1
    nright = 1;
end
nleft = length(size(X))-nright;
perm  = [(nleft+1):length(size(X)) (1:nleft)];
X = permute(X, perm);
end

%% Outerproudct of vectors
function X = GOuter(x)
X = x{1};
if length(x) > 1
    for i=2:length(x)
        if i==2
            X = GMP(X, x{i}',1);
        else
            X = GMP(X, x{i},0);
        end
    end
end
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

%% Centering function
function X = center(X)
X = X - mean(X,1);
end

%% Unfold array
function unfolded = unfold(X)
unfolded = reshape(X,size(X,1),[]);
end

%% Unfold transposed array
function unfolded = unfoldt(X)
X = shiftdim(X, ndims(X)-1);
unfolded = shiftdim(reshape(X,size(X,1),[]),1);
end

%% Vectorize
function vec = vec(X)
vec = X(:);
end

%% Vectorize and transpose
function vec = vect(X)
vec = X(:)';
end

%% ind2sub vector
function subidx = ind2subVec(sz, linidx)
tmp = linidx-1;
nd = max(length(sz),2);
subidx = zeros(1,nd);
for k=1:nd
    subk = mod(tmp, sz(k));
    subidx(:,k) = subk;
    tmp = (tmp-subk) / sz(k);
end
subidx = subidx+1;
end


