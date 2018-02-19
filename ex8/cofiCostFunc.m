function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));
for i = 1:num_movies
    x = X(i,:);
    idx = find(R(i,:) == 1);
    Theta_i = zeros(1, num_features);
    y = 0;
    if (isempty(idx) != 1)
        Theta_i = Theta(idx,:);
        y = Y(i, idx);
    end    
    diff = (x * Theta_i' - y);
    X_grad(i,:) =  diff * Theta_i + lambda * x;
    J = J + 1/2 * (diff * diff') + (lambda/2) * x * x';
 end

 for j = 1:num_users
    theta = Theta(j,:); % 1xf
    idx = (find(R(:, j) == 1))';
    x_j = zeros(1, num_features);
    y = 0;
    if (isempty(idx) != 1)
        x_j = X(idx,:); % idxSize x features 
        y = Y(idx, j)'; % 1 X idxSize
    end    
    diff = (theta * x_j' - y); 
    Theta_grad(j,:) = diff * x_j + lambda * theta;
    J = J + (lambda/2) * theta * theta';
end 


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
















% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
