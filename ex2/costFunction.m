function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
grad = zeros(size(theta));
prediction = sigmoid(X*theta);
grad  = (1/m)*X'*(prediction - y);
J = (1/m)*(-1*y'*log(prediction) -((1 - y)'*log(1 - prediction)));
end
