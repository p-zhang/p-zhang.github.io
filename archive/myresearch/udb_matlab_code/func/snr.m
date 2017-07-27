function x = snr(sig, ref)
% written by Junfeng Yang, Wotao Yin, and Yin Zhang

persistent ref_save N;
if nargin > 1; ref_save = ref; N = numel(ref); end;

mse = norm(ref_save-sig,'fro')^2/N;
if mse == 0; x = inf; return; end

dv = var(ref_save(:),1);
x = 10*log10(dv/mse);
