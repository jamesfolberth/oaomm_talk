function [] = rayleigh_grassman()

% Generate symmetric matrix
rng(271828);
n = 1000;
A = randn(n); A = 0.5*(A+A.');

% Create problem structure
p = 5;
M = grassmannfactory(n,p);
problem.M = M;

% Define the problem cost function and its Euclidean gradient
%problem.costgrad = @(Y) local_costgrad(M,A,Y);
problem.cost = @(Y) -trace(Y'*A*Y);
problem.grad = @(Y) -2*(A*Y - Y*(Y'*A*Y));

% Numerically check gradient consistency
%checkgradient(problem);

% Solve
opt = struct('tolgradnorm', 1e-6);
[Y,Ycost,info,opt] = trustregions(problem, [], opt);
%[Y,Ycost,info,opt] = steepestdescent(problem);

% Y is an ON representation of the invariant subspace corresponding to the p largest eigenvalues
[Vp,D] = eig(Y'*A*Y);    % gives the top p eigenvalues of A
diag(D)
eigs(A,p,'LA')
trace(Y'*A*Y)  % see Prop 2.1.1
sum(eigs(A,p,'LA'))

V = Y*Vp; % these are the p rightmost eigenvectors
V'*A*V

[Veigs,~] = eigs(A,p,'LA');
M.dist(Veigs,V) % induced distance

% Display some statistics
figure
semilogy([info.iter], [info.gradnorm], '.-');
title('Norm of the gradient of f');
xlabel('Iteration Number');
ylabel('Norm of the gradient of f');

end

function [f,g] = local_costgrad(M,A,Y)
   
   % This is unnecessary since grassmannfactory represents span(Y) as an ON matrix Y
   %[Q,~] = qr(Y,0); % Q = qf(Y) (this qr doesn't have R with positive diagonal, but
   %                 % it's span(Q) that matters, not the columns of Q)

   % see Sec 6.4.2 of AMS08
   f = -trace(Y'*(A*Y)); % pxp so trace is fine
   g = -2*(A*Y - Y*(Y'*(A*Y)));
end
