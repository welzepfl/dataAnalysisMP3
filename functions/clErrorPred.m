function [classError, classificationError] = clErrorPred(labels, prediction)

n_samples = length(labels);
n_positive = nnz(labels);
n_negative = nnz(labels==0);



    n_wpositive = sum(labels==0 & prediction==1);
    n_wnegative = sum(labels==1 & prediction==0);

classificationError = (n_wnegative+n_wpositive)/n_samples;
classError = 0.5*(n_wnegative/n_positive+n_wpositive/n_negative);