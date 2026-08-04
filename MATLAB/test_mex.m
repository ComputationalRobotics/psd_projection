% Smoke test for the MEX interface: compare every exposed method against
% MATLAB's own eig-based projection.
addpath('build');
n = 800;  rng(0);
A = randn(n);  A = (A + A')/2;

[P, D] = eig(A);  D = max(D, 0);  Aref = P*D*P';
nref = norm(Aref, 'fro');

[A1, r1] = psd_projection_MATLAB(A, 'adaptive', 1e-3);
fprintf('adaptive      : T=%d gemms=%d defl=%d pred=%.2e qual=%d | ACTUAL rel err=%.3e\n', ...
        r1.T, r1.gemms, r1.deflated, r1.predicted_rel_err, r1.qualified, norm(A1-Aref,'fro')/nref);

[A2, r2] = psd_projection_MATLAB(A, 'adaptive_FP16');
fprintf('adaptive_FP16 : T=%d gemms=%d defl=%d | ACTUAL rel err=%.3e\n', ...
        r2.T, r2.gemms, r2.deflated, norm(A2-Aref,'fro')/nref);

[A3, ev] = psd_projection_MATLAB(A, 'eig_FP64');
fprintf('eig_FP64      : rel err=%.3e | numel(eigenvalues)=%d min=%.3e\n', ...
        norm(A3-Aref,'fro')/nref, numel(ev), min(ev));

[~, r4] = psd_projection_MATLAB(A, 'adaptive', 1e-5);
fprintf('tol 1e-3 -> T=%d ; tol 1e-5 -> T=%d  (expect tighter tol >= T)\n', r1.T, r4.T);

% error handling
try
    psd_projection_MATLAB(A, 'nonsense');
    fprintf('ERROR: bad method was not rejected\n');
catch e
    fprintf('bad method correctly rejected: %s\n', e.message);
end
fprintf('MEX_TEST_OK\n');
