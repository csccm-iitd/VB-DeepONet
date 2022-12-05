clc
clear
close all

tic

%% %%

IFR = 0.95;
load train_PDE_ADVD.mat

n = 100000;
y = X_train0(1:n,:);

[~, sd, vd] = svd(y);

sd = sd.^2;
nd = 1;
chkd = sum(diag(sd));
rd = 0;
while rd < IFR
    rd = sum(diag(sd(1:nd,1:nd)))/chkd;
    nd = nd+1;
end
nd = nd-1;
fprintf('\n\n%d\n\n',nd);

red = y*vd(:,1:nd);

toc

%%

in = [red, X_train1(1:n, :)];
mdl = fitrgp(in, y_train(1:n,:));

toc

%%

if IFR == 0.95
    save("GPmdl95P_new_5.mat",'sd','vd','n','IFR','nd','mdl')
else
    save("GPmdl99d99P_new_5.mat",'sd','vd','n','IFR','nd','mdl')
end

%% PREDICTION

load test_PDE_ADVD.mat

S_mse = zeros(100,1);
S_nmse = zeros(100,1);

for i = 1:100

    i

    n = 10000;
    y = X_test0((i-1)*n+1:i*n, :);

    pfr = y*vd(:,1:nd);
    in = [pfr, X_test1((i-1)*n+1:i*n, :)];

    pred = zeros(10,10000,1);
    for j = 1:10
        pred(j,:,:) = predict(mdl, in);
    end

    mpred = squeeze(mean(pred, 1));
    spred = squeeze(std(pred, 1));

    mse = mean(mean((mpred'-y_test((i-1)*n+1:i*n,1)).^2));
    nmse = mean(mean((mpred'-y_test((i-1)*n+1:i*n,1)).^2))./mean(mean(y_test((i-1)*n+1:i*n,1).^2));

    S_mse(i) = mse;
    S_nmse(i) = nmse;
end

MSE = mean(S_mse)
NMSE = mean(S_nmse)

% MSE =
% 
%     0.0473
% 
% 
% NMSE =
% 
%     0.1146
