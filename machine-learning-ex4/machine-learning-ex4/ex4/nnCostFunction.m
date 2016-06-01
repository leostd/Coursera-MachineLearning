function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%Part 1:
Y = zeros(size(y,1), num_labels);
for i = 1:num_labels
    Y((y==i), i) = 1;
end

%h_0(x)
a1 = X;
a1 = [ones(size(y)) a1];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(y)) a2];
z3 = a2 * Theta2';
h0 = sigmoid(z3);

%Cost function
for i = 1:m
    for j = 1:num_labels
        J = J + Y(i,j)*log(h0(i,j)) + (1-Y(i,j))*log(1-h0(i,j));
    end
end
J = (-1/m) * J;

%Regularized
if lambda ~= 0
    aux1 = 0;
    aux2 = 0;
    Theta1Reg = Theta1(:,2:size(Theta1,2));
    Theta2Reg = Theta2(:,2:size(Theta2,2));
    for i = 1:hidden_layer_size
        for j = 1:input_layer_size
            aux1 = aux1 + Theta1Reg(i,j)^2;
        end
    end
    for i = 1:num_labels
        for j = 1:hidden_layer_size
            aux2 = aux2 + Theta2Reg(i,j)^2;
        end
    end
    
    aux = (aux1+aux2) * (lambda/(2*m));
    J = J + aux;
end

%Part 2
delta3 = h0 - Y;
for i = 1:m
    delta2 = Theta2(:,2:end)'*delta3(i,:)'.*sigmoidGradient(z2(i,:))';
    Theta2_grad = Theta2_grad + (delta3(i,:)' * a2(i,:));
    Theta1_grad = Theta1_grad + (delta2 * a1(i,:));
end
Theta2_grad = (1/m) * Theta2_grad;
Theta1_grad = (1/m) * Theta1_grad;


if lambda ~= 0
    Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + (lambda/m)*Theta2Reg;
    Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1Reg;
end









% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
