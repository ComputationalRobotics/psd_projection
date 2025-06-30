addpath("./build");

A = rand(10, 10); % generate a random matrix
A = A + A';  % make it symmetric

A_psd = psd_projection_MATLAB(A, 'composite_TF16'); % our method

% standard eigenvalue decomposition method
[P, D] = eig(A);
D = max(D, 0);
A_psd_eig = P * D * P';

% compare the results
disp(norm(A_psd - A_psd_eig, 'fro') / norm(A_psd_eig, 'fro'));