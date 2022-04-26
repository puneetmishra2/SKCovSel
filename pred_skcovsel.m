function [preds,r,rmsep] = pred_skcovsel(xt,yt,beta)
%for marking prediction based on skcovsel model calibrated on selected
%variables
nb = size(xt,2);

for i = 1:nb   % data mean centering 
    if length(size(xt{i}))>2
        dims{i} = size(xt{i});
        xt{i} = reshape(xt{i},dims{i}(1),prod(dims{i}(:,2:end)));
    end
end

xt = cell2mat(xt);
preds = [ones(size(xt,1),1) xt]*beta;

if exist('yt')
    for i = 1:size(yt,2)
        [r(1,i),rmsep(1,i)] = rmse(yt(:,i),preds(:,i));
        subplot(1,size(yt,2),i)
        plot(yt(:,i),preds(:,i),'or');xlabel('Measured');ylabel('Predicted');lsline;
        title(['R = ' num2str(round(r(1,i),2)) ' RMSEP = ' num2str(round(rmsep(1,i),2))]);
    end
else
    r = [];
    rmsep =[];
end

end


function [R,rms]=rmse(y,yhat)
R=corr(y(:),yhat(:));
rms=sqrt(sum((y(:)-yhat(:)).^2)/length(y));
end