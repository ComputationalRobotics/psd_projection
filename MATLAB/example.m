addpath("./build");

A = rand(5000, 5000); % generate a random matrix
A = A + A';  % make it symmetric

disp("start! \n");
A_psd = psd_projection_MATLAB(A, 'composite_TF16'); % our method
disp("end! \n");

% standard eigenvalue decomposition method
[P, D] = eig(A);
D = max(D, 0);
A_psd_eig = P * D * P';

% compare the results
disp(norm(A_psd - A_psd_eig, 'fro') / norm(A_psd_eig, 'fro'));