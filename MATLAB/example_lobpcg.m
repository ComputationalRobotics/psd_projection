% generate a random matrix and make it symmetric
addpath("./build");
n = 100;
A = randn(n, n);
A = (A + A') / 2;

m = 10;
[V, D] = sorteig(A);
V0 = V(:, 1:m);
D0 = D(1:m, 1:m);
D0 = diag(D0);
D0 = D0(:);
V0 = randn(n, m);

% % initial guess for the eigenvectors
% V0 = randn(n, m);
% % [V0, ~] = qr(V0, 'econ');
% D0 = randn(m, 1);

% call our LOBPCG method
[V, d_lobpcg] = lobpcg_MATLAB(A, V0, D0, m, true, 100, 1e-8, true);

% compare with MATLAB's built-in eigs function
[~, D_matlab] = eig(A);
d_matlab = diag(D_matlab);
d_matlab = sort(d_matlab, 'descend');
d_matlab = d_matlab(1:m);

% compare the eigenvalues
disp(norm(d_lobpcg - d_matlab) / norm(d_matlab));





function [V,D] = sorteig(A,order)
    if nargin < 2
        order = 'descend';
    end

    [V1,D1]     = eig(A);
    [~,idxsort] = sort(diag(D1),order);
    D           = D1(idxsort,idxsort);
    V           = V1(:,idxsort);
end