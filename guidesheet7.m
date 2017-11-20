% Miniproject III
% Guidesheet VII

% some initial code
addpath([pwd,'/functions']);
set(0,'DefaultAxesFontSize',14);
p_flag = 0; % exports figures only if set to 1

data = load('Data.mat');

% split data 0.7:0.3
[test, training, nTest, nTraining] = splitSet(data,0.7);

%% Regression X position

t_I = ones(size(training.PosX,1),1);
t_FM = training.Data;
X_train = [t_I t_FM];
t_I = ones(size(test.PosX,1),1);
t_FM = test.Data;
X_test = [t_I t_FM];

b_x = regress(training.PosX,X_train);
reg_train_x = X_train*b_x;
reg_test_x = X_test*b_x;
errX_train = immse(training.PosX,reg_train_x);
errX_test = immse(test.PosX,reg_test_x);

%% Regression Y position

t_I = ones(size(training.PosY,1),1);
t_FM = training.Data;
Y_train = [t_I t_FM];
t_I = ones(size(test.PosY,1),1);
t_FM = test.Data;
Y_test = [t_I t_FM];

b_y = regress(training.PosY,Y_train);
reg_train_y = Y_train*b_y;
reg_test_y = Y_test*b_y;
errY_train = immse(training.PosY,reg_train_y);
errY_test = immse(test.PosY,reg_test_y);


%% Plot

figure(99)
hold on
plot(training.PosX(1050:1100),training.PosY(1050:1100));
plot(reg_train_x(1050:1100),reg_train_y(1050:1100));
%plot(Y_train*b_x);
%plot(test.PosX);
%plot(Y_test*b_y);
legend('Original training','Regression training');%,'Original test','Regression test')

figure(98)
hold on
plot(test.PosX(1050:1100),test.PosY(1050:1100));
plot(reg_test_x(1050:1100),reg_test_y(1050:1100));
%plot(Y_train*b_x);
%plot(test.PosX);
%plot(Y_test*b_y);
legend('Original test','Regression test');%,'Original test','Regression test')


%% 2nd order Regression X position

t_I = ones(size(training.PosX,1),1);
t_FM = training.Data;
X_train = [t_I t_FM t_FM.^2];
t_I = ones(size(test.PosX,1),1);
t_FM = test.Data;
X_test = [t_I t_FM t_FM.^2];

b_x = regress(training.PosX,X_train);
reg_train_x = X_train*b_x;
reg_test_x = X_test*b_x;
errX2_train = immse(training.PosX,reg_train_x);
errX2_test = immse(test.PosX,reg_test_x);

%% 2nd order Regression Y position

t_I = ones(size(training.PosY,1),1);
t_FM = training.Data;
Y_train = [t_I t_FM t_FM.^2];
t_I = ones(size(test.PosY,1),1);
t_FM = test.Data;
Y_test = [t_I t_FM t_FM.^2];

b_y = regress(training.PosY,Y_train);
reg_train_y = Y_train*b_y;
reg_test_y = Y_test*b_y;
errY2_train = immse(training.PosY,reg_train_y);
errY2_test = immse(test.PosY,reg_test_y);


%% Plot

figure(97)
hold on
plot(training.PosX(1050:1100),training.PosY(1050:1100));
plot(reg_train_x(1050:1100),reg_train_y(1050:1100));
%plot(Y_train*b_x);
%plot(test.PosX);
%plot(Y_test*b_y);
legend('Original training','Regression training');%,'Original test','Regression test')

figure(96)
hold on
plot(test.PosX(1050:1100),test.PosY(1050:1100));
plot(reg_test_x(1050:1100),reg_test_y(1050:1100));
%plot(Y_train*b_x);
%plot(test.PosX);
%plot(Y_test*b_y);
legend('Original test','Regression test');%,'Original test','Regression test')




clear -regexp ^t_