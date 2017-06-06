function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
size(X);

J = sum((X*theta - y).^2)/(2*m) + sum(theta(2:end,:).^2)*lambda/(2*m);

reg=(lambda/m).*theta;
reg(1)=0;
#grad(1) = sum((X*theta - y).*X(:,1))/m;
grad = (X'*(X*theta - y))/m + reg;
#grad(1) = ((X*theta .- y)(1,:)'*X(:,1))/m;
#grad(2:end)=((X(:,2:end)*theta(2,:) .- y())'*X(:,2:end))/m + sum(sum(theta(2:end,:)))*lambda/m;
#grad(2)=((X(2:end,:)*theta(2,:) .- y(2,:))'*X(:,2:end))/m; + sum(sum(theta(2:end,:)))*lambda/m;





% =========================================================================

grad = grad(:);

end
