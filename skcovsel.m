function [Vars, crbvar,scores,beta] = skcovsel(x,y,lvs,mod,plotting,block_order)
% SKCOVSEL method for covariates selection in any type of data set
% Inputs :
% X: a cell with blocks, y: response (including multi-resposne)
% lvs: total variable to extract , mod: if only one three way array is
% input then possibility to define the modes to select variables
% plotting: 1 or 0
% block_order for sequential modelling such as [1 1 1 1 2 2 2 2]
xorig = x;
yorig = y;
nb = size(x,2);
dims = cell(1,nb);
vartot = zeros(1,nb);
crbvar = zeros(lvs,1);
scores = cell(1,lvs);
Q = cell(1,lvs);
R = cell(1,lvs);

for i = 1:nb   % data mean centering 
    if length(size(x{i}))>2
        cc = {[1 0 0; 0 0 0]};
        m_prepX=npreproc(x{i},cc{1});
        x{i} = m_prepX.Xprep;
        dims{i} = size(x{i});
        x{i} = reshape(x{i},dims{i}(1),prod(dims{i}(:,2:end)));
        vartot(1,i)=sum(sum(x{i}.*x{i})) ;
    else
        x{i} = x{i}-mean(x{i});
        dims{i} = size(x{i});
        vartot(1,i)=sum(sum(x{i}.*x{i})) ;
    end
end

y = y-mean(y); % data mean centering 
Vars = cell(4,lvs);
vartot(1,nb+1)= sum(sum(y.*y));

for a = 1:lvs % extraction of covariates
    all_scores = zeros(1,nb);
    temp_score = cell(1,nb);
    block_cc = zeros(1,nb);
    temp_Vars = cell(1,nb);
    temp_mode = cell(1,nb);
    temp_winner = cell(1,nb);
    temp_feature = cell(1,nb);
    for i = 1:nb % local ROCS run to select the best block in case of multiblock
        if length(dims{i})>2
            covar = Ra(x{i},y, y, []);
            covar = reshape(covar,dims{i}(:,2:end));
            [wjx,~,wkx] = svds(covar,1);
            if mod==0 % local ROCS competition to find the best mode for 3way arrat if mod is not defined
                loads = reshape(kron(wkx,wjx),dims{i}(:,2:end));
                temp_score_cov = max(max(abs(loads)));
                [indj,indk]=find(abs(loads)==temp_score_cov);
                temp_vecj = zeros(size(wjx,1),1);
                temp_vecj(indj,1)=1;
                temwjx = wjx.*temp_vecj;
                temp_veck = zeros(size(wkx,1),1);
                temp_veck(indk,1)=1;
                temwkx = wkx.*temp_veck ;
                ortho_scores = x{i}*kron(temwkx,temwjx);
                if a > 1,   ortho_scores = ortho_scores - cell2mat(scores(1:a-1))*(cell2mat(scores(1:a-1))'*ortho_scores); end
                [~,c] = Ra(ortho_scores,y, y, []);
                all_scores(1,1) = c;

                [~,indmj]  = max(abs(wjx));
                temp_vecj = zeros(size(wjx,1),1);
                temp_vecj(indmj,1)=1;
                temwjx = wjx.*temp_vecj;
                ortho_scores = x{i}*kron(temwkx,temwjx);
                if a > 1,   ortho_scores = ortho_scores - cell2mat(scores(1:a-1))*(cell2mat(scores(1:a-1))'*ortho_scores); end
                [~,c] = Ra(ortho_scores,y, y, []);
                all_scores(1,2) = c;

                [~,indmk]  = max(abs(wkx));
                temp_veck = zeros(size(wkx,1),1);
                temp_veck(indmk,1)=1;
                temwkx = wkx.*temp_veck ;
                ortho_scores = x{i}*kron(temwkx,temwjx);
                if a > 1,   ortho_scores = ortho_scores - cell2mat(scores(1:a-1))*(cell2mat(scores(1:a-1))'*ortho_scores); end
                [~,c] = Ra(ortho_scores,y, y, []);
                all_scores(1,3) = c;
            end

            if mod>0 % if mode is defined use the defined mode for extracting covariate for 3way data
                win = mod;
            else
                [~,win] = max(all_scores); % if mode not defined use the best mode from previous step
            end

            if win==1  % if mode 1 for 3way data
                loads = reshape(kron(wkx,wjx),dims{i}(:,2:end));
                temp_score_cov = max(max(abs(loads)));
                [indj,indk]=find(abs(loads)==temp_score_cov);
                temp_vecj = zeros(size(wjx,1),1);
                temp_vecj(indj,1)=1;
                temwjx = wjx.*temp_vecj;
                temp_veck = zeros(size(wkx,1),1);
                temp_veck(indk,1)=1;
                temwkx = wkx.*temp_veck ;
                ortho_scores = x{i}*kron(temwkx,temwjx);
                if a > 1,   ortho_scores = ortho_scores - cell2mat(scores(1:a-1))*(cell2mat(scores(1:a-1))'*ortho_scores); end
                ortho_scores  = ortho_scores /norm(ortho_scores);
                if nb>1;[~,block_cc(1,i)] = Ra(ortho_scores,y, y, []);end
                temp_score{1,i} = ortho_scores;
                temp_Vars{1,i} = [indj,indk];
                temp_mode{1,i} = 'Mode 1';
                temp_winner{1,i} = ['Block ' num2str(i)];
                temp_feature{1,i} = reshape(kron(temwkx,temwjx),dims{i}(:,2:end));
            elseif win==2 % if mode 2 for 3way data
                [~,indmj]  = max(abs(wjx));
                temp_vecj = zeros(size(wjx,1),1);
                temp_vecj(indmj,1)=1;
                temwjx = wjx.*temp_vecj;
                ortho_scores = x{i}*kron(wkx,temwjx);
                if a > 1,   ortho_scores = ortho_scores - cell2mat(scores(1:a-1))*(cell2mat(scores(1:a-1))'*ortho_scores); end
                ortho_scores  = ortho_scores /norm(ortho_scores);
                loads_for_reg(:,a) = kron(wkx,temwjx);
                if nb>1;[~,block_cc(1,i)] = Ra(ortho_scores,y, y, []);end
                temp_score{1,i} = ortho_scores;
                temp_Vars{1,i} = indmj;
                temp_mode{1,i} = 'Mode 2';
                temp_winner{1,i} = ['Block ' num2str(i)];
                temp_feature{1,i} = reshape(kron(wkx,temwjx),dims{i}(:,2:end));
            elseif win==3 % if mode 3 for 3way data
                [~,indmk]  = max(abs(wkx));
                temp_veck = zeros(size(wkx,1),1);
                temp_veck(indmk,1)=1;
                temwkx = wkx.*temp_veck;
                ortho_scores = x{i}*kron(temwkx,wjx);
                if a > 1,   ortho_scores = ortho_scores - cell2mat(scores(1:a-1))*(cell2mat(scores(1:a-1))'*ortho_scores); end
                ortho_scores  = ortho_scores /norm(ortho_scores);
                loads_for_reg(:,a) = kron(temwkx,wjx);
                if nb>1;[~,block_cc(1,i)] = Ra(ortho_scores,y, y, []);end
                temp_score{1,i} = ortho_scores;
                temp_Vars{1,i} = indmk;
                temp_mode{1,i} = 'Mode 3';
                temp_winner{1,i} = ['Block ' num2str(i)];
                temp_feature{1,i} = reshape(kron(temwkx,wjx),dims{i}(:,2:end));
            end
        else  %when block is just a 2-way it becomes fCovSel (or same solution as CovSel)

            [~,indmj] = max( sum( (x{i}'*y).^2 ,2) );
            temp_veck = zeros(size(x{i},2),1);
            temp_veck(indmj,1)=1;
            temp_v = temp_veck;
            temp_feature{1,i} = temp_v;
            ortho_scores = x{i}*temp_v;
            if a > 1,   ortho_scores = ortho_scores - cell2mat(scores(1:a-1))*(cell2mat(scores(1:a-1))'*ortho_scores); end
            ortho_scores  = ortho_scores /norm(ortho_scores);
            if nb>1;[~,block_cc(1,i)] = Ra(ortho_scores,y, y, []);end
            temp_score{1,i} = ortho_scores;
            temp_Vars{1,i} = indmj;
            temp_mode{1,i} = 'Mode 2';
            temp_winner{1,i} = ['Block ' num2str(i)];
            [~,block_cc(1,i)] = Ra(x{i}*temp_v,y, y, []);
        end
    end

    if block_order(1,1)>0
        winner = block_order(1,a);
    else
        if nb>1
            [~,winner] = max(block_cc);
        elseif nb==1
            winner = 1;
        end
    end

    blocks_selected(1,a) = winner;
    scores{1,a} = temp_score{1,winner};
    R{1,a} = y;
    Q{1,a} = y'*scores{1,a};
    y = y - scores{1,a}*(scores{1,a}'*y);
    crbvar(a,1) = (vartot(1,nb+1) - sum(sum(y.*y)) ) / vartot(1,nb+1);
    Vars{1,a} = temp_Vars{1,winner};
    Vars{2,a} = temp_mode{1,winner};
    Vars{3,a} = temp_winner{1,winner};
    Vars{4,a} = temp_feature{1,winner};

end

%%%%% post-processing for regression coefficients %%%%%%%%%%%
for_offset = [];
scores = cell2mat(scores);
Q = cell2mat(Q);
for i = 1:size(x,2)
    temp_data =[];
    modes = cell2mat(Vars(2,:)');
    modes(:,1:5)=[];
    modes = str2num(modes);
    if length(size(xorig{i}))==2
        temp_vars = cell2mat(Vars(1,blocks_selected==i));
        if sum(temp_vars)==0
            all_weights{i} =[];
        else
            for_offset = [for_offset xorig{i}(:,temp_vars)];
            temp_data = xorig{i}(:,temp_vars);
            all_weights{i} = eye(size(temp_vars,2));
        end
    else
        temp_modes = modes(blocks_selected==i);
        if sum(temp_modes)==0
            all_weights{i} =[];
        else
            if sum(temp_modes==1)>0
                temp_vars = Vars(1,blocks_selected==i);
                temp_vars = temp_vars(1,temp_modes==1);
                temp_vars = cell2mat(temp_vars');
                for jo = 1:size(temp_vars,1)
                    temp_data(:,jo) = xorig{i}(:,temp_vars(jo,1),temp_vars(jo,2));
                end
                for_offset = [for_offset temp_data];
                all_weights{i} = eye(size(temp_data,2));
            end
            if sum(temp_modes==2)>0
                temp_vars = Vars(1,blocks_selected==i);
                loads_for_reg = loads_for_reg(:,blocks_selected==i);
                temp_vars = temp_vars(1,temp_modes==2);
                temp_vars = cell2mat(temp_vars');
                temp_data = xorig{i}(:,temp_vars,:);
                temp_dims = size(temp_data);
                temp_data = reshape(temp_data,[temp_dims(1) temp_dims(2)*temp_dims(3)]);
                for_offset = [for_offset temp_data];
                temp_W = loads_for_reg(~all(loads_for_reg == 0, 2),:);
                temp_W(temp_W==0)=[];
                temp_W = reshape(temp_W,temp_dims(3),size(temp_vars,1));
                W = [];
                for lo = 1:size(temp_W,1)
                    W = [W;eye(size(temp_vars,1)).*temp_W(lo,:)];
                end
                all_weights{i} = W;
            end
            if sum(temp_modes==3)>0
                temp_vars = Vars(1,blocks_selected==i);
                loads_for_reg = loads_for_reg(:,blocks_selected==i);
                temp_vars = temp_vars(1,temp_modes==3);
                temp_vars = cell2mat(temp_vars');
                temp_data = xorig{i}(:,:,temp_vars);
                temp_dims = size(temp_data);
                if length(size(temp_data))==2
                    for_offset = [for_offset temp_data];
                    temp_W = loads_for_reg(~all(loads_for_reg == 0, 2),:);
                    temp_W(temp_W==0)=[];
                    temp_W = reshape(temp_W,temp_dims(2),size(temp_vars,1));
                    W = [];
                    for lo = 1:size(temp_W,1)
                        W = [W;eye(size(temp_vars,1)).*temp_W(lo,:)];
                    end
                    all_weights{i} = W;
                elseif length(unique(temp_vars))==1
                    for_offset = [for_offset temp_data];
                    temp_W = loads_for_reg(~all(loads_for_reg == 0, 2),:);
                    temp_W(temp_W==0)=[];
                    temp_W = reshape(temp_W,temp_dims(2),size(temp_vars,1));
                    W = [];
                    for lo = 1:size(temp_W,1)
                        W = [W;eye(size(temp_vars,1)).*temp_W(lo,:)];
                    end
                    all_weights{i} = W;
                else
                    temp_data = reshape(temp_data,[temp_dims(1) temp_dims(2)*temp_dims(3)]);
                    for_offset = [for_offset temp_data];
                    temp_W = loads_for_reg(~all(loads_for_reg == 0, 2),:);
                    temp_W(temp_W==0)=[];
                    temp_W = temp_W';
                    temp_W = repmat(temp_W,1,size(temp_vars,1));
                    temp_eye = eye(size(temp_vars,1));
                    W = [];
                    for lo = 1:size(temp_vars,1)
                        temp_vec = repmat(temp_eye(lo,:),temp_dims(2),1);
                        temp_vec = reshape(temp_vec,[temp_dims(2)*size(temp_vars,1) 1]);
                        W = [W temp_W(:,lo).*temp_vec];
                    end
                    all_weights{i} = W;
                end
            end
        end
    end
end


Pb = for_offset'*scores; % X-loadings
W = [];
temp_lv = 0;
eyes = eye(lvs);
for i = 1:size(all_weights,2)
    if ~isempty(all_weights{1,i})
        if length(size(xorig{i}))==2
            if i==1
                W =[W;eyes(1:size(all_weights{1,i},2),:)];
                temp_lv = temp_lv + size(all_weights{1,i},2);
            else
                W =[W;eyes(temp_lv+1:temp_lv+size(all_weights{1,i},2),:)];
                temp_lv = temp_lv + size(all_weights{1,i},2);
            end
        elseif length(size(xorig{i}))==3
            if i==1
                if isdiag(all_weights{1,i})
                    W =[W;eyes(1:size(all_weights{1,i},2),:)];
                    temp_lv = temp_lv + size(all_weights{1,i},2);
                else
                    W =[W;[all_weights{1,i} zeros(size(all_weights{1,i},1),lvs-size(all_weights{1,i},2))]];
                    temp_lv = temp_lv + size(all_weights{1,i},2);
                end
            else
                if isdiag(all_weights{1,i})
                    W =[W;eyes(temp_lv+1:temp_lv+size(all_weights{1,i},2),:)];
                    temp_lv = temp_lv + size(all_weights{1,i},2);
                else
                    W =[W; [zeros(size(all_weights{1,i},1),temp_lv)   all_weights{1,i}  zeros(size(all_weights{1,i},1),lvs-temp_lv-size(all_weights{1,i},2))]];
                    temp_lv = temp_lv + size(all_weights{1,i},2);
                end
            end
        end
    end
end

% regression coefficients estimation
for t = 1:size(Q,1)
    temp_beta = cumsum(bsxfun(@times,W/triu(Pb'*W), Q(t,:)),2);
    temp_beta = [mean(yorig(:,t)) - mean(for_offset)*temp_beta(:,end); temp_beta(:,end)];
    beta(:,t) = temp_beta;
end
ypred = [ones(size(for_offset,1),1) for_offset]*beta;

%%%%%%%%% plotting %%%%%%%%%%%%%%%%%%%
if plotting==1
    figure,
    subplot(1,2,2);plot(crbvar*100,'-o');xlabel('Variables selected');ylabel('Explained variance (%)');
    legend('Y variance');
    axis tight;
    subplot(1,2,1)
    tk = cell2mat(Vars(3,:)');
    tk = tk(:,end);
    plot(str2num(tk),'-or');xlabel('Selected variables');ylabel('Block order');
    axis tight;
    figure,
    for i = 1:size(beta,2)
        subplot(1,size(beta,2),i)
        plot(yorig(:,i),ypred(:,i),'o');xlabel('Measured');ylabel('Predicted');lsline;
        [r,rm]=rmse(yorig(:,i),ypred(:,i));
        title(['R = ' num2str(round(r,2)) ' RMSEP = ' num2str(round(rm,2))]);
    end
end
end

%%%%%%%%  auxillary functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function model=npreproc(X,preproc,varargin)
if isempty(varargin)
    scalefact='mse';
else
    scalefact=varargin{1};
end

% storing the dimensions of the array
dimX=size(X);
preppars=cell(size(preproc));

Xp=X;

%loop across the different modes ('scale first')

for i=1:3
    pord=[i:3 1:i-1];

    Xp=permute(Xp, pord);
    dimp=dimX(pord);


    if preproc(2,i)==1
        xu=reshape(Xp,dimp(1), dimp(2)*dimp(3));
        switch scalefact
            case 'std'
                s=std(xu, [],2);
            case 'mse'
                s=sqrt(sum(xu.^2,2)./(dimp(2)*dimp(3)));
        end

        xp=xu./repmat(s,1,dimp(2)*dimp(3));
        Xp=reshape(xp,dimp(1),dimp(2),dimp(3));
        preppars{2,i}=s;
    end

    Xp=ipermute(Xp, pord);

end


for i=1:3
    pord=[i:3 1:i-1];
    Xp=permute(Xp, pord);
    dimp=dimX(pord);


    if preproc(1,i)==1
        xu=reshape(Xp,dimp(1),dimp(2)*dimp(3));
        m=mean(xu);
        xp=xu-repmat(m,dimp(1),1);
        Xp=reshape(xp,dimp(1),dimp(2),dimp(3));
        preppars{1,i}=m;
    end
    Xp=ipermute(Xp, pord);

end




model.Xprep=Xp;
model.Xraw=X;
model.prepropt=preproc;
model.preppars=preppars;
model.scalefact=scalefact;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%% Ra function (for CPLS excluding powers)
function [w,cc] = Ra(X, Y, Yprim, wt)
W = X'*Y;
[r,A] = ccXY(X*W, Yprim, wt); % Computaion of canonical correlations between
% XW and Y with the rows weighted according to wt.
w     = W*A(:,1);             % The optimal loading weight vector
cc    = r(1)^2;               % squared canonical correlation
end

%%%%%%%%%%%%%%%%%%%% Canonical correlations
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

%%%%%%%%%%%%%%%%% Weighted centering
function [X, mX, n, p] = Center(X,wt)
% Centering of the data matrix X by subtracting the weighted column means
% according to the nonegative weights wt
[n,p] = size(X);
% Calculation of column means:
if nargin == 2 && ~isempty(wt)
    mX = (wt'*X)./sum(wt);
else
    mX = mean(X);
end
% Centering of X, similar to: %X = X-ones(n,1)*mX;
X = X-repmat(mX,n,1);
end

function X = rscale(X,d)
% Scaling the rows of the matrix X by the values of the vector d
X = repmat(d,1,size(X,2)).*X;
end

function [R,rms]=rmse(y,yhat)
R=corr(y(:),yhat(:));
rms=sqrt(sum((y(:)-yhat(:)).^2)/length(y));
end
