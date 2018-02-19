function [alpha] = learnLearningRate(X, y, theta, num_iters)
% Choose some alpha value
alphaInitial = 0.03;
alphaFinal = alphaInitial/(3.0^5);
alphaStep = 1.0/3.0;
firstIteration = 0;
J_minPrevious = Inf;
alpha = 0.0;
for alphaCurrent = alphaInitial:alphaStep:alphaFinal
  % Init Theta and Run Gradient Descent 
  theta = zeros(3, 1);
  [theta, J_history] = gradientDescentMulti(X, y, theta, alphaCurrent, num_iters);
  J_min_current = min(J_history);
  if (J_min_current <= J_minPrevious)
    J_minPrevious = J_min_current;
  else
    alpha = alphaCurrent;
    break;
  endif  
  % Plot the convergence graph
  % find minimum of J
  %figure;
  semilogy(1:50, J_history(1:50), 'color', rand(1,3), 'LineWidth', 2);
  if(firstIteration == 0)
    hold on;
    xlabel('Number of iterations');
    ylabel('Cost J');
    firstIteration = 1;
  endif
end
end