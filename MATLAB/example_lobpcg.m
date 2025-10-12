% generate a random matrix and make it symmetric
addpath("./build");
n = 100;
A = randn(n, n);
A = (A + A') / 2;

% initial guess for the eigenvectors
m = 10; % number of eigenvalues to compute
V0 = randn(m, n);
D0 = randn(m, 1);

% call our LOBPCG method
[V, d_lobpcg] = lobpcg_MATLAB(A, V0, D0, m, true, 100, 1e-8, true);

% compare with MATLAB's built-in eigs function
[~, D_matlab] = eig(A);
d_matlab = diag(D_matlab);
d_matlab = sort(d_matlab, 'descend');
d_matlab = d_matlab(1:m);

% compare the eigenvalues
disp(norm(d_lobpcg - d_matlab) / norm(d_matlab));