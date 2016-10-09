% Following the tutorial example http://www.manopt.org/tutorial.html
function [] = rayleigh_sphere()

% Generate symmetric matrix
rng(271828);
n = 1000;
A = randn(n); A = 0.5*(A+A.');

% Create problem structure
M = spherefactory(n);
problem.M = M;

% Define the problem cost function and its Euclidean gradient
%problem.costgrad = @(x) local_costgrad(M,A,x);
problem.cost = @(x) -x'*(A*x);
problem.egrad = @(x) -2*A*x;
problem.ehess = @(x,u) -2*A*u;

% Numerically check gradient consistency
%checkgradient(problem);
%checkhessian(problem);

% Solve
[x,xcost,info,opt] = trustregions(problem);
%[x,xcost,info,opt] = steepestdescent(problem);

% Display some statistics
figure
semilogy([info.iter], [info.gradnorm], '.-');
xlabel('Iteration Number');
ylabel('Norm of the gradient of f');

end

function [f,g] = local_costgrad(M,A,x)
   
   Ax = A*x;
   f = -x'*Ax;
   g = M.egrad2rgrad(x,-2*Ax);

end
