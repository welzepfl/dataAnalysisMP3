% Miniproject III
% Guidesheet 8

% some initial code
addpath([pwd,'/functions']);
set(0,'DefaultAxesFontSize',14);
p_flag = 0; % exports figures only if set to 1

data = load('Data.mat');

dataRegression = 0;
dataLasso = 0;
dataElasticNets = 0;

% split data 0.05:0.95
[test, training, nTest, nTraining] = splitSet(data,0.05);

%% Regression

if (~dataRegression)
    I_train = ones(size(training.PosX,1),1);
    t_FM = training.Data;
    X_train_lin = [I_train t_FM];
    I_test = ones(size(test.PosX,1),1);
    t_FM = test.Data;
    X_test_lin = [I_test t_FM];
    
    % X position
    b_x_lin = regress(training.PosX,X_train_lin);
    reg_train_x_lin = X_train_lin*b_x_lin;
    reg_test_x_lin = X_test_lin*b_x_lin;
    errX_train_lin = immse(training.PosX,reg_train_x_lin);
    errX_test_lin = immse(test.PosX,reg_test_x_lin);

    % Y position
    b_y_lin = regress(training.PosY,X_train_lin);
    reg_train_y_lin = X_train_lin*b_y_lin;
    reg_test_y_lin = X_test_lin*b_y_lin;
    errY_train_lin = immse(training.PosY,reg_train_y_lin);
    errY_test_lin = immse(test.PosY,reg_test_y_lin);
    
    save('dataRegression.mat','b_x_lin','b_y_lin','reg_train_x_lin','reg_train_y_lin',...
        'reg_test_x_lin','reg_test_y_lin','errX_train_lin','errY_train_lin',...
        'errX_test_lin','errY_test_lin');
else
    load('dataRegression.mat');
end
    

%% Plot

figure(99)
hold on
t_interval = 150:200;
plot(training.PosX(t_interval),training.PosY(t_interval));
plot(reg_train_x_lin(t_interval),reg_train_y_lin(t_interval));
xlabel('X position of joystick');
ylabel('Y position of joystick');
legend('Original','Linear regression');%,'Original test','Regression test')
if p_flag
    print('figures/LinRegPlotTraining05','-dpng');
    print('figures/LinRegPlotTraining05','-depsc');
end

figure(98)
t_interval = 1050:1100;
hold on
plot(test.PosX(t_interval),test.PosY(t_interval));
plot(reg_test_x_lin(t_interval),reg_test_y_lin(t_interval));
xlabel('X position of joystick');
ylabel('Y position of joystick');
legend('Original','Linear regression');%,'Original test','Regression test')
if p_flag
    print('figures/LinRegPlotTest05','-dpng');
    print('figures/LinRegPlotTest05','-depsc');
end

%% LASSO

if (~dataLasso)
    kFold = 10;
    lambda = logspace(-10,0,15);

    % lasso estimation for x and y positions including 10 fold cross-
    % validation
    [b_x_lasso, fitInfo_x_lasso] = lasso(training.Data,training.PosX,'lambda',lambda,'cv',kFold); % X position
    [b_y_lasso, fitInfo_y_lasso] = lasso(training.Data,training.PosY,'lambda',lambda,'cv',kFold); % Y position
    
    % number of non zero elements in beta vector
    for indLambda = 1:size(b_x_lasso,2)
        nnz_x_lasso(indLambda) = nnz(b_x_lasso(:,indLambda));
        nnz_y_lasso(indLambda) = nnz(b_y_lasso(:,indLambda));
    end
    
    if (~(fitInfo_x_lasso.IndexMinMSE == fitInfo_y_lasso.IndexMinMSE))
        disp('index of min error is not the same for x as for y')
    end
    
    lasso_x_test = fitInfo_x_lasso.Intercept(fitInfo_x_lasso.IndexMinMSE)+...
            test.Data*b_x_lasso(:,fitInfo_x_lasso.IndexMinMSE);
    lasso_y_test = fitInfo_y_lasso.Intercept(fitInfo_y_lasso.IndexMinMSE)+...
            test.Data*b_y_lasso(:,fitInfo_y_lasso.IndexMinMSE);
    err_x_test_lasso = immse(test.PosX,lasso_x_test);
    err_y_test_lasso = immse(test.PosY,lasso_y_test);
    
    save('dataLasso.mat','b_x_lasso','fitInfo_x_lasso','b_y_lasso','fitInfo_y_lasso',...
        'nnz_x_lasso','nnz_y_lasso','lasso_x_test','lasso_y_test','err_x_test',...
        'err_y_test','lambda');
else
    load('dataLasso.mat');
end
    
%% Plot
    
figure(97) % Mean square error as function of lambda
semilogx(lambda,fitInfo_x_lasso.MSE);
hold on
semilogx(lambda,fitInfo_y_lasso.MSE);
xlabel('\lambda');
ylabel('MSE');
legend('x','y');
if p_flag
    print('figures/MSE_lasso','-dpng');
    print('figures/MSE_lasso','-depsc');
end

figure(96)
semilogx(lambda,nnz_x_lasso)
hold on
semilogx(lambda,nnz_y_lasso)
legend('x','y')
xlabel('\lambda')
ylabel('Number of non-zero elements in \beta')
if p_flag
    print('figures/nnz_lasso','-dpng');
    print('figures/nnz_lasso','-depsc');
end

figure(95)
t_interval = 1050:1100;
hold on
plot(test.PosX(t_interval),test.PosY(t_interval))
plot(lasso_x_test(t_interval),lasso_y_test(t_interval))
legend('Original data','Lasso estimation');
xlabel('X position of joystick');
ylabel('Y position of joystick');
if p_flag
    print('figure/lasso_pos','-dpng')
    print('figure/lasso_pos','-depsc')
end

%% Elastic nets

if (~dataElasticNets)
    kFold = 10;
    alpha = 0.5; % elastic nets coefficient
    
    % elastic nets estimation of x and y positions including 10 fold
    % cross-validation
    [b_x_en, fitInfo_x_en] = lasso(training.Data,training.PosX,'lambda',lambda,'cv',kFold,'Alpha',alpha);
    [b_y_en, fitInfo_y_en] = lasso(training.Data,training.PosY,'lambda',lambda,'cv',kFold,'Alpha',alpha);
    
    % number of non zero elements in beta vector
    for indLambda = 1:size(b_x_en,2)
        nnz_x_en(indLambda) = nnz(b_x_en(:,indLambda));
        nnz_y_en(indLambda) = nnz(b_y_en(:,indLambda));
    end
    
    if (~(fitInfo_x_lasso.IndexMinMSE == fitInfo_y_lasso.IndexMinMSE))
        disp('index of min error is not the same for x as for y')
    end

    en_x_test = fitInfo_x_en.Intercept(fitInfo_x_en.IndexMinMSE)+...
            test.Data*b_x_en(:,fitInfo_x_en.IndexMinMSE);
    en_y_test = fitInfo_y_en.Intercept(fitInfo_y_en.IndexMinMSE)+...
            test.Data*b_y_en(:,fitInfo_y_en.IndexMinMSE);
    err_x_test_en = immse(test.PosX,en_x_test);
    err_y_test_en = immse(test.PosY,en_y_test);
else
    load('dataElasticNets')
end

%% Plot
    
figure(97) % Mean square error as function of lambda
hold on
semilogx(lambda,fitInfo_x_en.MSE);
semilogx(lambda,fitInfo_y_en.MSE);
xlabel('\lambda');
ylabel('MSE');
legend('x lasso','y lasso','x elastic nets','y elastic nets');
if p_flag
    print('figures/MSE_lassoEN','-dpng');
    print('figures/MSE_lassoEN','-depsc');
end

figure(96)
hold on
semilogx(lambda,nnz_x_en)
semilogx(lambda,nnz_y_en)
legend('x lasso','y lasso','x elastic nets','y elastic nets');
xlabel('\lambda')
ylabel('Number of non-zero elements in \beta')
if p_flag
    print('figures/nnz_lassoEN','-dpng');
    print('figures/nnz_lassoEN','-depsc');
end

figure(94)
t_interval = 1050:1100;
hold on
plot(test.PosX(t_interval),test.PosY(t_interval))
plot(en_x_test(t_interval),en_y_test(t_interval))
legend('Original data','Lasso estimation');
xlabel('X position of joystick');
ylabel('Y position of joystick');
if p_flag
    print('figure/en_pos','-dpng')
    print('figure/en_pos','-depsc')
end
