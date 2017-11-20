function [testSet, trainingSet, n_training, n_test] = splitSet(dataSet, ratioTraining)

trainingSet = struct('features',0,'labels',0);
testSet = struct('features',0,'labels',0);
randomOrderDataSet = struct('features',0,'labels',0);

n_training = round(size(dataSet.labels,1)*ratioTraining);
n_test = length(dataSet.labels)-n_training;

permutedVect = randperm(length(dataSet.labels));
randomOrderDataSet.features = dataSet.features(permutedVect,:);
randomOrderDataSet.labels = dataSet.labels(permutedVect,:);

trainingSet.features = randomOrderDataSet.features(1:n_training,:);
trainingSet.labels = randomOrderDataSet.labels(1:n_training);

testSet.features = randomOrderDataSet.features(n_training+1:end,:);
testSet.labels = randomOrderDataSet.labels(n_training+1:end);