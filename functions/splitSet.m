function [testSet, trainingSet, n_test, n_training] = splitSet(data, ratioTraining)

trainingSet = struct('Data',0,'PosX',0,'PosY',0);
testSet = struct('Data',0,'PosX',0,'PosY',0);

n_training = round(length(data.PosX)*ratioTraining);
n_test = length(data.PosX)-n_training;

trainingSet.Data = data.Data(1:n_training,:);
trainingSet.PosX = data.PosX(1:n_training);
trainingSet.PosY = data.PosY(1:n_training);

testSet.Data = data.Data(n_training+1:end,:);
testSet.PosX = data.PosX(n_training+1:end);
testSet.PosY = data.PosY(n_training+1:end);

