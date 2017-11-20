function [classError, classificationError] = clError(feature, labels, threshold)

n_samples = length(labels);
n_positive = nnz(labels);
n_negative = nnz(labels==0);

n_wpositive = nnz(feature(labels==0)>threshold);
n_wnegative = nnz(feature(labels==1)<threshold);

classificationError = (n_wnegative+n_wpositive)/n_samples;
classError = 0.5*(n_wnegative/n_positive+n_wpositive/n_negative);