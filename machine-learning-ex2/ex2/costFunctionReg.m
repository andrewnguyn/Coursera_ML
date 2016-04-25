function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% SIZES:
% hypot: 118x1
% theta: 28x1
% grad: 28x1
% X: 118x28
% y: 118x1
% lambda: 1x1
hypot = sigmoid(X*theta);

% grad1 = (1/m) * (hypot - y)' {118x1}' * (X(:,1)) {118x1};
grad(1) = (1/m) * (X(:,1))' * (hypot - y);

% grad(2:end) = (1/m) * (hypot - y)' {1x118} * X(:,2:end) {118x27} + (lambda/m) * theta(2:end) {27x1});
% {27x1} = {27x1} + {27x1}
grad(2:end) = (1/m) * ( X(:,2:end)' * (hypot - y) ) + (lambda/m) * theta(2:end);

% J = (1/m) * ( -y .* log(hypot) - (1-y) .* log(1 - hypot)) {118x1} + (lambda/(2*m)) * theta.^2 ) {28x1};
J = (1/m) * ( -y' * log(hypot) - (1-y)' * log(1 - hypot)) + (lambda/(2*m)) * sum(theta(2:end).^2) ;

% =============================================================

end
