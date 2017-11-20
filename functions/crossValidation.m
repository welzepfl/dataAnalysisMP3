function [idealNbrOfFeatures,validationError,trainingError,classifierArray] = crossValidation(features,labels,rankingType,classifierType,kFold)
%CROSSVALIDATION - calculates a cross validation on the dataSet
%   Input: features: NxM - N data samples with M features
%          labels: Nx1 - labels (1,0)
%          rankingType: string, accepted: 'fisher','corr'
%          classifierType: string, accepted:
%          'linear','diaglinear','quadratic','diagquadratic'
%          kFold: integer -> maximum N-1

%% INPUT VALIDATION

switch nargin
    case 2
        rankingType = 'fisher';
        classifierType = 'linear';
        kfold = 10;
        print('Default: fisher,linear,k=10');
    case 3
        classifierType = 'linear';
        kfold = 10;
    case 4
        kfold = 10;
    case 1
        print('need more than 1 argument');
        return
    otherwise
        print('Check you number of arguments. Stupid');
end

if size(features,1) ~= size(labels,1)
    print('features and labels do not have same size')
    return
end
if ~strcmp(rankingType,'fisher') && ~strcmp(rankingType,'corr')
    print('wrong ranking type');
    return
end

switch classifierType
    case 'diaglinear'
    case 'diagquadratic'
    case 'linear'
    case 'quadratic'
    otherwise
        print('Non-valid classifier type')
        return
end

if ~isinteger(kFold)
    print('kfold must be integer');
    return
end

if kfold >= size(features,2)
    print('kfold too big');
    return
end

%% FUNCTION BODY

% create data partition
cpL = cvpartition(labels, 'KFold',kFold);


% Feature Number Loop
for idxFeatures = 1:size(features,2)
    
    % Validation Loop
    for idxKfold = 1:kFold
        % do the classifier
        classifierArray{idxFeatures,idxKFold} = fitcdiscr(features(cpL.training(idxKfold),1:idxFeatures),...
            labels(cpL.training(idxKfold)),'DiscrimType',classifierType); % temporary Classifier to calculate error -> not stored
        
        % train errors
        trainingLabels = predict(classifierArray{idxFeatures,idxKfold},features(cpL.training(idxKfold),1:idxFeatures));
        trainingError(idxFeatures,idxKfold) = clErrorPred(labels(cpL.training(idxKfold)),trainingLabels);
        % validation errors
        validationLabels = predict(classifierArray{idxFeatures,idxKfold},features(cpL.test(idxKfold),1:idxFeatures)); % validation label prediction
        validationError(idxFeatures,idxInnerFold,idxOuterFold) = clErrorPred(labels(cpL.test(idxKfold)),validationLabels);
    end
end

averageValidationError = mean(validationError,2);
[~,idealNbrOfFeatures] = min(averageValidationError);
end

