function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

pass = find(y==1);
fail = find(y == 0);
plot(X(pass, 1), X(pass, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);    
plot(X(fail, 1), X(fail, 2), 'ko','MarkerFaceColor', 'y', 'MarkerSize', 7);    

hold off;

end
