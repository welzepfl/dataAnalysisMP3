% Miniproject III
% Guidesheet 8
% Final model, no CV

% some initial code
close all
clear
addpath([pwd,'/functions']);
set(0,'DefaultAxesFontSize',14);
p_flag = 0; % exports figures only if set to 1

data = load('Data.mat');

% Idea:
% create training, and test set
% implement in training set in order to optimize
% hyperparameters (lambda, alpha, number of PCs)
% 
% TODO: number of PC's

% split data training & validation: 0.7, test: 0.3
[test, train_val, nTest, nTrainVal] = splitSet(data,0.7);
[validation, training, nVal, nTraining] = splitSet(train_val,0.7);

% PCA
[coeff, trainingPCA, variance] = pca(training.Data);
test_centered = test.Data-(mean(training.Data,1));
testPCA = test_centered*coeff;
val_centered = validation.Data-(mean(training.Data,1));
validationPCA = val_centered*coeff;

% normalize PCA
normTrainingPCA = trainingPCA./mean(trainingPCA);
normTestPCA = testPCA./mean(trainingPCA);
normValidationPCA = validationPCA./mean(trainingPCA);

% define range of optimization vectors
n_lambda = 15;
lambda = logspace(-10,0,n_lambda);
n_alpha = 16;
alpha = linspace(0.01,1,n_alpha);
n_fold = 10;
nStep = 40; % step of features increased per loop

% check if previous file exists
if exist('beta_ss.mat','file')
    disp('Datafile found')
    load('beta_ss.mat');
    countPC = idxPC+1;
else
    countPC = 1;
    b_x = zeros(size(normTrainingPCA,2)/nStep,n_alpha,n_lambda,size(normTrainingPCA,2));
    b_y = zeros(size(normTrainingPCA,2)/nStep,n_alpha,n_lambda,size(normTrainingPCA,2));
end

% k-fold cross-validation % nope: takes too much time. We have enough
% samples
for idxPC = countPC:(size(normTrainingPCA,2)/nStep)
    tic;
    for idxAlpha = 1:n_alpha
        for idxLambda = 1:n_lambda
            [b_x(idxPC,idxAlpha,idxLambda,1:idxPC*nStep),...
                fitInfo_x(idxPC,idxAlpha,idxLambda)] =...
                lasso(normTrainingPCA(:,1:idxPC*nStep),training.PosX,...
                'Lambda',lambda(idxLambda),'Alpha',alpha(idxAlpha));
            [b_y(idxPC,idxAlpha,idxLambda,1:idxPC*nStep),...
                fitInfo_y(idxPC,idxAlpha,idxLambda)] =...
                lasso(normTrainingPCA(:,1:idxPC*nStep),training.PosY,...
                'Lambda',lambda(idxLambda),'Alpha',alpha(idxAlpha));
            t_val_pos_x = fitInfo_x(idxPC,idxAlpha,idxLambda).Intercept+...
                normValidationPCA*reshape(b_x(idxPC,idxAlpha,idxLambda,:),[960 1]);
            t_val_pos_y = fitInfo_y(idxPC,idxAlpha,idxLambda).Intercept+...
                normValidationPCA*reshape(b_y(idxPC,idxAlpha,idxLambda,:),[960 1]);
            fitInfo_x(idxPC,idxAlpha,idxLambda).MSE = immse(validation.PosX, t_val_pos_x);
            fitInfo_y(idxPC,idxAlpha,idxLambda).MSE = immse(validation.PosY, t_val_pos_y);
        end
        % This part of the code is not optimized in terms of efficiency
        [t_minLambda_x(idxAlpha), t_indLambda_x(idxAlpha)] = min([fitInfo_x(idxPC,idxAlpha,:).MSE]);
        [t_minLambda_y(idxAlpha), t_indLambda_y(idxAlpha)] = min([fitInfo_y(idxPC,idxAlpha,:).MSE]);
    end
    % find minimum MSE for each number of PCs
    [minAlpha_x(idxPC), indAlpha_x(idxPC)] = min(t_minLambda_x);
    [minAlpha_y(idxPC), indAlpha_y(idxPC)] = min(t_minLambda_y);
    indLambda_x(idxPC) = t_indLambda_x(indAlpha_x(idxPC));
    indLambda_y(idxPC) = t_indLambda_y(indAlpha_y(idxPC));
    
    % checkpoint
    save('beta_ss.mat','b_x','b_y','minAlpha_x','minAlpha_y','indLambda_x',...
        'indLambda_y','indAlpha_x','indAlpha_y','fitInfo_x','fitInfo_y','idxPC');
    t_loop = toc;
    disp(['saved after ',num2str(idxPC*nStep),' PCs after ',num2str(t_loop),' s'])
end

%% Plot

figure(99)
plot(minAlpha_x)
xlabel('Number of principal components')
ylabel('MSE')
xticks([3:3:24]);
xticklabels({'120','240','360','480','600','720','840','960'})
grid on
if p_flag
    print('figure/mse_ss_x','-dpng')
    print('figure/mse_ss_x','-depsc')
end

figure(98)
plot(minAlpha_y)
xlabel('Number of principal components')
ylabel('MSE')
xticks([3:3:24]);
xticklabels({'120','240','360','480','600','720','840','960'})
grid on
if p_flag
    print('figure/mse_ss_y','-dpng')
    print('figure/mse_ss_y','-depsc')
end

figure(97)
plot(indAlpha_x)
xlabel('Number of principal components')
ylabel('Ideal alpha value')
xticks([3:3:24]);
xticklabels({'120','240','360','480','600','720','840','960'})
yticks([2:2:n_alpha]);
yticklabels({num2str(alpha(2)),num2str(alpha(4)),num2str(alpha(6)),...
    num2str(alpha(8)),num2str(alpha(10)),num2str(alpha(12)),...
    num2str(alpha(14)),num2str(alpha(16))});
grid on
if p_flag
    print('figure/idealAlpha_ss_x','-dpng')
    print('figure/idealAlpha_ss_x','-depsc')
end

figure(96)
plot(indAlpha_y)
xlabel('Number of principal components')
ylabel('Ideal alpha value')
xticks([3:3:24]);
xticklabels({'120','240','360','480','600','720','840','960'})
yticks([2:2:n_alpha]);
yticklabels({num2str(alpha(2)),num2str(alpha(4)),num2str(alpha(6)),...
    num2str(alpha(8)),num2str(alpha(10)),num2str(alpha(12)),...
    num2str(alpha(14)),num2str(alpha(16))});
grid on
if p_flag
    print('figure/idealAlpha_ss_y','-dpng')
    print('figure/idealAlpha_ss_y','-depsc')
end

%% Evaluate final model

nPC = 440;
lambda_x = fitInfo_x(nPC/nStep,indAlpha_x(nPC/nStep),indLambda_x(nPC/nStep)).Lambda;
lambda_y = fitInfo_y(nPC/nStep,indAlpha_y(nPC/nStep),indLambda_y(nPC/nStep)).Lambda;
alpha_x = fitInfo_x(nPC/nStep,indAlpha_x(nPC/nStep),indLambda_x(nPC/nStep)).Alpha;
alpha_y = fitInfo_y(nPC/nStep,indAlpha_y(nPC/nStep),indLambda_y(nPC/nStep)).Alpha;

normTrainValPCA = cat(1,normTrainingPCA,normValidationPCA);

[betaX, fitInfoX] = lasso(normTrainValPCA(:,1:nPC),train_val.PosX,'Lambda',lambda_x,'Alpha',alpha_x);
[betaY, fitInfoY] = lasso(normTrainValPCA(:,1:nPC),train_val.PosY,'Lambda',lambda_y,'Alpha',alpha_y);

test_x = fitInfoX.Intercept+normTestPCA(:,1:nPC)*betaX;
test_y = fitInfoY.Intercept+normTestPCA(:,1:nPC)*betaY;
test_err_x = immse(test.PosX, test_x);
test_err_y = immse(test.PosY, test_y);

