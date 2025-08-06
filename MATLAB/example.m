addpath("./build");

A = rand(5000, 5000); % generate a random matrix
A = A + A';  % make it symmetric

A_psd = psd_projection_MATLAB(A, 'composite_FP32'); % our method
% [A_psd, eigenvalues] = psd_projection_MATLAB(A, 'eig_FP64'); % cuSOLVER factorization

% standard eigenvalue decomposition method
[P, D] = eig(A);
D = max(D, 0);
A_psd_eig = P * D * P';

% compare the results
disp(norm(A_psd - A_psd_eig, 'fro') / norm(A_psd_eig, 'fro'));