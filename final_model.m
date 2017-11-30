% Miniproject III
% Guidesheet 8
% Final model

% some initial code
close all
clear
addpath([pwd,'/functions']);
set(0,'DefaultAxesFontSize',14);
p_flag = 0; % exports figures only if set to 1

data = load('Data.mat');

% Idea:
% create training, and test set
% implement (10-fold) cross-validation in training set in order to optimize
% hyperparameters (lambda, alpha, number of PCs)
% 
% TODO: number of PC's

% split data 0.7:0.3
[test, training, nTest, nTraining] = splitSet(data,0.7);

% PCA
[coeff, trainingPCA, variance] = pca(training.Data);
test_centered = test.Data-(mean(training.Data,1));
testPCA = test_centered*coeff;

% normalize PCA
nTrainingPCA = trainingPCA./mean(trainingPCA);
nTestPCA = testPCA./mean(trainingPCA);

% define range of optimization vectors
n_lambda = 15;
lambda = logspace(-10,0,n_lambda);
n_alpha = 16;
alpha = linspace(0.01,1,n_alpha);
n_fold = 10;
nStep = 60; % step of features increased per loop

% check if previous file exists
if exist('beta.mat','file')
    disp('Datafile found')
    load('beta.mat');
    countPC = idxPC+1;
else
    countPC = 1;
    b_x = zeros(size(nTrainingPCA,2)/nStep,n_alpha,n_lambda,size(nTrainingPCA,2));
    b_y = zeros(size(nTrainingPCA,2)/nStep,n_alpha,n_lambda,size(nTrainingPCA,2));
end

% k-fold cross-validation % nope: takes too much time. We have enough
% samples
for idxPC = countPC:(size(nTrainingPCA,2)/nStep)
    tic;
    for idxAlpha = 1:n_alpha
        for idxLambda = 1:n_lambda
            [b_x(idxPC,idxAlpha,idxLambda,1:idxPC*nStep), fitInfo_x(idxPC,idxAlpha,idxLambda)] = lasso(nTrainingPCA(:,1:idxPC*nStep),training.PosX,'CV',n_fold,'lambda',lambda(idxLambda),'Alpha',alpha(idxAlpha));
            [b_y(idxPC,idxAlpha,idxLambda,1:idxPC*nStep), fitInfo_y(idxPC,idxAlpha,idxLambda)] = lasso(nTrainingPCA(:,1:idxPC*nStep),training.PosY,'CV',n_fold,'lambda',lambda(idxLambda),'Alpha',alpha(idxAlpha));
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
    save('beta.mat','b_x','b_y','minAlpha_x','minAlpha_y','indLambda_x',...
        'indLambda_y','indAlpha_x','indAlpha_y','fitInfo_x','fitInfo_y','idxPC');
    t_loop = toc;
    disp(['saved after ',num2str(idxPC*nStep),' PCs after ',num2str(t_loop),' s'])
end

figure(99)
plot([1:16],minAlpha_x)
xlabel('Number of principal components')
ylabel('MSE')
xticks([2:2:16]);
xticklabels({'120','240','360','480','600','720','840','960'})
grid on


figure(98)
plot([1:16],minAlpha_y)
xlabel('Number of principal components')
ylabel('MSE')
grid on

figure(97)
plot([1:16],indAlpha_x)
xlabel('Number of principal components')
ylabel('Ideal alpha value')
yticks([2:2:n_alpha]);
yticklabels({num2str(alpha(2)),num2str(alpha(4)),num2str(alpha(6)),...
    num2str(alpha(8)),num2str(alpha(10)),num2str(alpha(12)),...
    num2str(alpha(14)),num2str(alpha(16))});
grid on


% test_x = fitInfo_x(indAlpha_x,indLambda_x(indAlpha_x)).Intercept+...
%     nTestPCA*reshape(b_x(indAlpha_x,indLambda_x(indAlpha_x),:),[960 1]);
% test_y = fitInfo_y(indAlpha_y,indLambda_y(indAlpha_y)).Intercept+...
%     nTestPCA*reshape(b_y(indAlpha_y,indLambda_y(indAlpha_y),:),[960,1]);
% test_err_x = immse(test.PosX, test_x);
% test_err_y = immse(test.PosY, test_y);






