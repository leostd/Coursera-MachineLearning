function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;
C_vec = [0.01 0.03 0.1 0.3 1 3 10 30];
Sigma_vec = [0.01 0.03 0.1 0.3 1 3 10 30];
error = 0;
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

errorMin = inf;
a = 0; 
b = 0;
for i=1:length(C_vec)
    for j=1:length(Sigma_vec)
        svm = svmTrain(X, y, C_vec(i),@(x1, x2) gaussianKernel(x1, x2, Sigma_vec(j)));
        predictions = svmPredict(svm, Xval);
        error = mean(double(predictions ~= yval));
        if( error<errorMin )
            a = i;
            b = j;
            errorMin = error;
        end
    end
end
C = C_vec(a);
sigma = Sigma_vec(b);








% =========================================================================

end
