function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
J = 0;
J_LC = 0;
J_Reg = 0;
grad = zeros(size(theta));
gradLC = zeros(size(theta));
gradReg = zeros(size(theta));
thetaReg = zeros(size(theta));
scale = ones(size(theta));
scale(1) = 0.0;
thetaReg = theta .*scale;

prediction = sigmoid(X*theta);
gradLC  = (1/m)*X'*(prediction - y);
gradReg = (lambda/m) * thetaReg;
J_LC = (1/m)*(-1*y'*log(prediction) -((1 - y)'*log(1 - prediction)));
J_Reg = (lambda/(2*m)) * (thetaReg' * thetaReg);
J = J_LC + J_Reg;
grad  = gradLC + gradReg;
end
