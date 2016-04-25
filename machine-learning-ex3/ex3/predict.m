function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% SIZES
% Theta1  = 25x401
% Theta2 = 10x26
% X = 5000x401
% m = 5000
% num_labels = 10
% p = 5000x1


% add bias unit for X
X = [ones(m,1), X];

% layer2 {5000x25} = X {5000x401} * Theta1' {401x25} = {5000x25}
% layer2 {5000x25} = {5000x25}
z2 = X * Theta1';
layer2 = sigmoid(z2);

% add bias unit for layer2
% layer2 {500x26} = [ones(), layer2]; {500x26}
layer2 = [ones(m,1), layer2];

% layer3 = layer2 {500x26} * Theta2' {26x10};
% layer3 {500x10}
z3 = layer2 * Theta2';
layer3 = sigmoid(z3);

[m_val, idx] = max(layer3');

p = idx;




% =========================================================================


end
