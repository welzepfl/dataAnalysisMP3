% Miniproject III
% Guidesheet 8
% Final model

% some initial code
addpath([pwd,'/functions']);
set(0,'DefaultAxesFontSize',14);
p_flag = 0; % exports figures only if set to 1

data = load('Data.mat');

% Idea:
% create training, and test set
% implement (10-fold) cross-validation in training set in order to optimize
% hyperparameters (lambda, alpha, number of features/PC's)
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

b_x = zeros(n_alpha,n_lambda,size(nTrainingPCA,2)); b_y = zeros(n_alpha,n_lambda,size(nTrainingPCA,2));

% k-fold cross-validation
%for idxFold = 1:n_fold
    for idxAlpha = 1:n_alpha
        for idxLambda = 1:n_lambda
%             [b_x(idxAlpha,idxLambda,:), fitInfo_x(idxAlpha,idxLambda)] = lasso(nTrainingPCA,training.PosX,'CV',n_fold,'lambda',lambda(idxLambda),'Alpha',alpha(idxAlpha));
%             [b_y(idxAlpha,idxLambda,:), fitInfo_y(idxAlpha,idxLambda)] = lasso(nTrainingPCA,training.PosY,'CV',n_fold,'lambda',lambda(idxLambda),'Alpha',alpha(idxAlpha));
        end
        % This part of the code is not optimized in terms of efficiency
        [minLambda_x(idxAlpha), indLambda_x(idxAlpha)] = min([fitInfo_x(idxAlpha,:).MSE]);
        [minLambda_y(idxAlpha), indLambda_y(idxAlpha)] = min([fitInfo_y(idxAlpha,:).MSE]);
    end
%end

[minAlpha_x, indAlpha_x] = min(minLambda_x);
[minAlpha_y, indAlpha_y] = min(minLambda_y);

test_x = fitInfo_x(indAlpha_x,indLambda_x(indAlpha_x)).Intercept+...
    nTestPCA*reshape(b_x(indAlpha_x,indLambda_x(indAlpha_x),:),[960 1]);
test_y = fitInfo_y(indAlpha_y,indLambda_y(indAlpha_y)).Intercept+...
    nTestPCA*reshape(b_y(indAlpha_y,indLambda_y(indAlpha_y),:),[960,1]);
test_err_x = immse(test.PosX, test_x);
test_err_y = immse(test.PosY, test_y);






