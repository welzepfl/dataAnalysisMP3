% Miniproject III
% Guidesheet VII

% some initial code
addpath([pwd,'/functions']);
set(0,'DefaultAxesFontSize',14);
p_flag = 0; % exports figures only if set to 1

data = load('Data.mat');

dataPCA    = 1;
dataLin    = 1;
data2order = 1;
dataNfeat  = 1;

% split data 0.7:0.3
[test, training, nTest, nTraining] = splitSet(data,0.7);

%% PCA

if (~dataPCA)
    [coeffs, trainPCA, variance] = pca(training.Data);
    t_testData = test.Data-(mean(training.Data,1));
    testPCA = t_testData*coeffs;
    
    save('dataPCA.mat','trainPCA','testPCA');
else
    load('dataPCA.mat');
end

% Plot
figure(94)
plot(cumsum(variance)/sum(variance))
hold on
plot([1,960],[0.9,0.9],'--')
xlabel('Number of PCs')
ylabel('Relative cumulative cariance')
xlim([0,960])
if p_flag
    print('figures/cumsum','-dpng')
    print('figures/cumsum','-depsc')
end

%% Normalization
t_nTraining = training.Data./mean(training.Data);
test.Data = test.Data./mean(training.Data);
training.Data = t_nTraining;

%% Linear regression 
% X position

if (~dataLin)
    I_train = ones(size(training.PosX,1),1);
    t_FM = training.Data;
    X_train_lin = [I_train t_FM];
    I_test = ones(size(test.PosX,1),1);
    t_FM = test.Data;
    X_test_lin = [I_test t_FM];

    b_x_lin = regress(training.PosX,X_train_lin);
    reg_train_x_lin = X_train_lin*b_x_lin;
    reg_test_x_lin = X_test_lin*b_x_lin;
    errX_train_lin = immse(training.PosX,reg_train_x_lin);
    errX_test_lin = immse(test.PosX,reg_test_x_lin);

    % Y position

    %I_train = ones(size(training.PosY,1),1);
    t_FM = training.Data;
    Y_train_lin = [I_train t_FM];
    %I_test = ones(size(test.PosY,1),1);
    t_FM = test.Data;
    Y_test_lin = [I_test t_FM];

    b_y_lin = regress(training.PosY,Y_train_lin);
    reg_train_y_lin = Y_train_lin*b_y_lin;
    reg_test_y_lin = Y_test_lin*b_y_lin;
    errY_train_lin = immse(training.PosY,reg_train_y_lin);
    errY_test_lin = immse(test.PosY,reg_test_y_lin);
    
    save('dataLin.mat','b_x_lin','b_y_lin','reg_train_x_lin','reg_train_y_lin',...
        'reg_test_x_lin','reg_test_y_lin','errX_train_lin','errY_train_lin',...
        'errX_test_lin','errY_test_lin');
else
    load('dataLin.mat');
end

%% Plot

figure(99)
hold on
plot(training.PosX(1050:1100),training.PosY(1050:1100));
plot(reg_train_x_lin(1050:1100),reg_train_y_lin(1050:1100));
xlabel('X position of joystick');
ylabel('Y position of joystick');
legend('Original','Linear regression');%,'Original test','Regression test')
if p_flag
    print('figures/LinRegPlotTraining','-dpng');
    print('figures/LinRegPlotTraining','-depsc');
end

figure(98)
hold on
plot(test.PosX(1050:1100),test.PosY(1050:1100));
plot(reg_test_x_lin(1050:1100),reg_test_y_lin(1050:1100));
xlabel('X position of joystick');
ylabel('Y position of joystick');
legend('Original','Linear regression');%,'Original test','Regression test')
if p_flag
    print('figures/LinRegPlotTest','-dpng');
    print('figures/LinRegPlotTest','-depsc');
end

%% 2nd order Regression X position

if (~data2order)
    I_train = ones(size(training.PosX,1),1);
    t_FM = training.Data;
    X_train_2 = [I_train t_FM t_FM.^2];
    I_test = ones(size(test.PosX,1),1);
    t_FM = test.Data;
    X_test_2 = [I_test t_FM t_FM.^2];

    b_x_2 = regress(training.PosX,X_train_2);
    reg_train_x_2 = X_train_2*b_x_2;
    reg_test_x_2 = X_test_2*b_x_2;
    errX_train_2 = immse(training.PosX,reg_train_x_2);
    errX_test_2 = immse(test.PosX,reg_test_x_2);

    % Y position

    %I_train = ones(size(training.PosY,1),1);
    t_FM = training.Data;
    Y_train_2 = [I_train t_FM t_FM.^2];
    %I_test = ones(size(test.PosY,1),1);
    t_FM = test.Data;
    Y_test_2 = [I_test t_FM t_FM.^2];

    b_y_2 = regress(training.PosY,Y_train_2);
    reg_train_y_2 = Y_train_2*b_y_2;
    reg_test_y_2 = Y_test_2*b_y_2;
    errY_train_2 = immse(training.PosY,reg_train_y_2);
    errY_test_2 = immse(test.PosY,reg_test_y_2);

    save('data2order.mat','b_x_2','b_y_2','reg_train_x_2','reg_train_y_2',...
        'reg_test_x_2','reg_test_y_2','errX_train_2','errY_train_2',...
        'errX_test_2','errY_test_2');
else
    load('data2order.mat');
end

%% Plot

figure(97)
hold on
plot(training.PosX(1050:1100),training.PosY(1050:1100));
plot(reg_train_x_2(1050:1100),reg_train_y_2(1050:1100));
xlabel('X position of joystick');
ylabel('Y position of joystick');
legend('Original','2nd order regression');%,'Original test','Regression test')
if p_flag
    print('figures/2regPlotTrain','-dpng');
    print('figures/2regPlotTrain','-depsc');
end

figure(96)
hold on
plot(test.PosX(1050:1100),test.PosY(1050:1100));
plot(reg_test_x_2(1050:1100),reg_test_y_2(1050:1100));
xlabel('X position of joystick');
ylabel('Y position of joystick');
legend('Original','2nd order regression');%,'Original test','Regression test')
if p_flag
    print('figures/2regPlotTest','-dpng');
    print('figures/2regPlotTest','-depsc');
end

%% Increase number of features

if (~dataNfeat)
    nStep = 60; % defines the number of features added with each iteration
    h = size(data.Data,2)/nStep;
    I_train = ones(size(training.PosX,1),1);
    I_test = ones(size(test.PosX,1),1);
    
    %Preallocating variables
    err_train_xlin = zeros(h,1);   err_train_x2 = zeros(h,1);
    err_train_ylin = zeros(h,1);   err_train_y2 = zeros(h,1);
    err_test_xlin = zeros(h,1);    err_test_x2 = zeros(h,1);
    err_test_ylin = zeros(h,1);    err_test_y2 = zeros(h,1);
    
    % Initializing variables
    X_train = I_train; X_test = I_test;
    X_train2 = I_train; X_test2 = I_test;
    
    % Loop
    for i=1:h
        t_FM_train = training.Data(:,nStep*(i-1)+1:nStep*i);
        t_FM_test = test.Data(:,nStep*(i-1)+1:nStep*i);
        X_train = [X_train t_FM_train];%(:,end-nStep+1:end)]; % simply appends the last 60 elements to the already existing vector
        X_test = [X_test t_FM_test];%(:,end-nStep+1:end)];
        X_train2 = [X_train, X_train2(:,nStep*(i-1)+1:end), t_FM_train.^2];%(:,end-nStep+1:end).^2]; % avoid having to recalculate all the squares again
        X_test2 = [X_test, X_test2(:,nStep*(i-1)+1:end), t_FM_test.^2];%(:,end-nStep+1:end).^2];
        b_xlin = regress(training.PosX,X_train);
        b_x2 = regress(training.PosX,X_train2);
        b_ylin = regress(training.PosY,X_train);
        b_y2 = regress(training.PosY,X_train2);
        reg_train_xlin = X_train*b_xlin;
        reg_train_x2 = X_train2*b_x2;
        reg_train_ylin = X_train*b_ylin;
        reg_train_y2 = X_train2*b_y2;
        reg_test_xlin = X_test*b_xlin;
        reg_test_x2 = X_test2*b_x2;
        reg_test_ylin = X_test*b_ylin;
        reg_test_y2 = X_test2*b_y2;
        err_train_xlin(i) = immse(training.PosX,reg_train_xlin);
        err_train_x2(i) = immse(training.PosX,reg_train_x2);
        err_train_ylin(i) = immse(training.PosY,reg_train_ylin);
        err_train_y2(i) = immse(training.PosY,reg_train_y2);
        err_test_xlin(i) = immse(test.PosX,reg_test_xlin);
        err_test_x2(i) = immse(test.PosX,reg_test_x2);
        err_test_ylin(i) = immse(test.PosY,reg_test_ylin);
        err_test_y2(i) = immse(test.PosY,reg_test_y2);
    end
    
    err_train_lin = (err_train_xlin+err_train_ylin)/2;
    err_train_2 = (err_train_x2+err_train_y2)/2;
    err_test_lin = (err_test_xlin+err_test_ylin)/2;
    err_test_2 = (err_test_x2+err_test_y2)/2;
    
    save('dataNfeat.mat','err_train_lin','err_train_2','err_test_lin','err_test_2','h','nStep');
else
    load('dataNfeat.mat');
end


figure(95)
plot(1:h,err_train_lin,1:h,err_train_2,1:h,err_test_lin,1:h,err_test_2)
legend('linear training','2nd order training','linear test','2nd order test')
xticks([2:2:h])
xticklabels(string(2*nStep:2*nStep:size(data.Data,2)))
xlabel('Number of features')
ylabel('Error')
grid on
if p_flag
    print('functions/errNfeat','-dpng')
    print('functions/errNfeat','-depsc')
end

clear -regexp ^t_