addpath("./build");

A = rand(5000, 5000); % generate a random matrix
A = A + A';  % make it symmetric

% our method: the spectrum-aware adaptive projection.  The matrix does NOT
% need to be pre-scaled, and the optional third argument is the target
% relative Frobenius error (default 1e-3).
[A_psd, report] = psd_projection_MATLAB(A, 'adaptive', 1e-3);
% A_psd = psd_projection_MATLAB(A, 'adaptive_FP16');      % half precision
% [A_psd, eigenvalues] = psd_projection_MATLAB(A, 'eig_FP64'); % cuSOLVER factorization

fprintf('chose T=%d (%d GEMMs), deflated %d eigenpairs, predicted rel err %.2e\n', ...
        report.T, report.gemms, report.deflated, report.predicted_rel_err);

% standard eigenvalue decomposition method
[P, D] = eig(A);
D = max(D, 0);
A_psd_eig = P * D * P';

% compare the results
disp(norm(A_psd - A_psd_eig, 'fro') / norm(A_psd_eig, 'fro'));
