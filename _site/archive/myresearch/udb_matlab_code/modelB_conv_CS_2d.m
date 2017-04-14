%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation results for the structured sensing model proposed in our paper
%        
%        'Modulated Unit-Norm Tight Frames for Compressed Sensing'
%
%      Section IV.B. Convolutional CS with deterministic phase modulation
%                     
% To compare the performance between the proposed convolutional based CS model with
% existing convolutional based CS models by subsampling and recovering an image.
%
% Measruement Matrix (\Phi):
% 1) Rancom convolution
% 2) Partial random circulant matrix 
% 3) Random convolution with extended Golay sequence
% 4) Partial random circulant with Golay modulation
%
% Sparsifying Basis (\Psi) and Recovery Algorithm: SARA package
%
%
% Simulation takes about 3 minutes on a computer with:
% Processor: Intel(R) Core(TM) i7-3770 CPU @ 3.4 GHz 3.4 GHz
% Installed memory (RAM): 8.00 GB
% System type: 64-bit Operating System
% Softwar: MATLAB R2012a (7.14.0.739)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


close all;
clear all;

%% Clear workspace

clc
clear;
close all
clear all

%% Define paths

addpath misc/
addpath prox_operators/
addpath test_images/



%% Read image

imagename = 'lena.pgm';
%imagename = 'portofino.pgm';

% Load image
im = im2double(imread(imagename));

% Normalise
im = im/max(max(im));

% Enforce positivity
im(im<0) = 0;

%% Parameters

input_snr = 30; % Noise level (on the measurements)

%Undersampling ratio M/N
p=0.25;


%% Sparsity operators

%Wavelet decomposition depth
nlevel=4;

dwtmode('per');
[C,S]=wavedec2(im,nlevel,'db8'); 
ncoef=length(C);
[C1,S1]=wavedec2(im,nlevel,'db1'); 
ncoef1=length(C1);
[C2,S2]=wavedec2(im,nlevel,'db2'); 
ncoef2=length(C2);
[C3,S3]=wavedec2(im,nlevel,'db3'); 
ncoef3=length(C3);
[C4,S4]=wavedec2(im,nlevel,'db4'); 
ncoef4=length(C4);
[C5,S5]=wavedec2(im,nlevel,'db5'); 
ncoef5=length(C5);
[C6,S6]=wavedec2(im,nlevel,'db6'); 
ncoef6=length(C6);
[C7,S7]=wavedec2(im,nlevel,'db7'); 
ncoef7=length(C7);

%SARA

Psit = @(x) [wavedec2(x,nlevel,'db1')'; wavedec2(x,nlevel,'db2')';wavedec2(x,nlevel,'db3')';...
    wavedec2(x,nlevel,'db4')'; wavedec2(x,nlevel,'db5')'; wavedec2(x,nlevel,'db6')';...
    wavedec2(x,nlevel,'db7')';wavedec2(x,nlevel,'db8')']/sqrt(8); 

Psi = @(x) (waverec2(x(1:ncoef1),S1,'db1')+waverec2(x(ncoef1+1:ncoef1+ncoef2),S2,'db2')+...
    waverec2(x(ncoef1+ncoef2+1:ncoef1+ncoef2+ncoef3),S3,'db3')+...
    waverec2(x(ncoef1+ncoef2+ncoef3+1:ncoef1+ncoef2+ncoef3+ncoef4),S4,'db4')+...
    waverec2(x(ncoef1+ncoef2+ncoef3+ncoef4+1:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5),S5,'db5')+...
    waverec2(x(ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+1:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6),S6,'db6')+...
    waverec2(x(ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+1:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+ncoef7),S7,'db7')+...
    waverec2(x(ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+ncoef7+1:ncoef1+ncoef2+ncoef3+ncoef4+ncoef5+ncoef6+ncoef7+ncoef),S,'db8'))/sqrt(8);


%%
% Mask
mask = rand(size(im)) < p; 
ind = find(mask==1);
% Masking matrix (sparse matrix in matlab)
Ma = sparse(1:numel(ind), ind, ones(numel(ind), 1), numel(ind), numel(im));
figure,
spy(Ma)
title('the random mask')
% Masking matrix with fixed position
Mafx = sparse(1:numel(ind),1:numel(ind),ones(numel(ind), 1),numel(ind), numel(im));
figure,
spy(Mafx)
title('the fixed mask')
% the random (diagonal) matrix    
ss=rand(size(im));
D=(2*(ss<0.5)-1);

% the golay modulation vector and matrix
[g1,g2]=generate_golay(log(numel(im))/log(2));
G = reshape(g1,size(im));

% the extended golay (diagonal) matrix
[g3,g4]=generate_golay(log(numel(im)/2)/log(2));
g5 = zeros(1,numel(im));
g5(1:numel(im)/2)=g3;
g5(numel(im)/2+1)=g3(1);

g5(numel(im)/2+2:numel(im))=g3(numel(im)-(numel(im)/2+2:numel(im))+2);
Gex = reshape(g5,size(im));

% A = @(x) Ma*reshape(fft2(D.*x)/sqrt(numel(ind)), numel(x), 1);
%At = @(x) D.*(ifft2(reshape(Ma'*x(:), size(im))*sqrt(numel(ind))));
%%
% randcom convolution operator
ss=rand(size(im));
D=(2*(ss<0.5)-1);
A = @(x) Ma*reshape(ifft2(D.*fft2(x)),numel(x),1);
At = @(x) ifft2(fft2(reshape(Ma'*x(:), size(im))).*D);

% partial random convolution operator
A2 = @(x) Mafx*reshape(ifft2(D.*fft2(x)),numel(x),1);
At2 = @(x) ifft2(fft2(reshape(Mafx'*x(:), size(im))).*D);

% random convolution with extended golay sequence
A4 = @(x) Ma*reshape(ifft2(Gex.*fft2(x)),numel(x),1);
At4 = @(x) ifft2(fft2(reshape(Ma'*x(:), size(im))).*Gex);

% partial random convolution with golay modulation
A3 = @(x) Mafx*reshape(ifft2(D.*fft2(x.*G)),numel(x),1);
At3 = @(x) G.*ifft2(fft2(reshape(Mafx'*x(:), size(im))).*D);
%%
% Sampling
y = A(im);
% Add Gaussian i.i.d. noise
sigma_noise = 10^(-input_snr/20)*std(im(:));
y = y + (randn(size(y)) + 1i*randn(size(y)))*sigma_noise/sqrt(2);

% Sampling
y2 = A2(im);
% Add Gaussian i.i.d. noise
y2 = y2 + (randn(size(y2)) + 1i*randn(size(y2)))*sigma_noise/sqrt(2);

% Sampling
y3 = A3(im);
% Add Gaussian i.i.d. noise
y3 = y3 + (randn(size(y3)) + 1i*randn(size(y3)))*sigma_noise/sqrt(2);

% Sampling
y4 = A4(im);
% Add Gaussian i.i.d. noise
y4 = y4 + (randn(size(y4)) + 1i*randn(size(y4)))*sigma_noise/sqrt(2);

% Tolerance on noise
epsilon = sqrt(numel(y)+2*sqrt(numel(y)))*sigma_noise;
epsilon_up = sqrt(numel(y)+2.1*sqrt(numel(y)))*sigma_noise;
    
%%    
% Parameters for BPDN
param.verbose = 1; % Print log or not
param.gamma = 1e-1; % Converge parameter
param.rel_obj = 5e-4; % Stopping criterion for the L1 problem
param.max_iter = 200; % Max. number of iterations for the L1 problem
param.nu_B2 = 1; % Bound on the norm of the operator A
param.tol_B2 = 1-(epsilon/epsilon_up); % Tolerance for the projection onto the L2-ball
param.tight_B2 = 0; % Indicate if A is a tight frame (1) or not (0)
param.pos_B2 = 1; %Positivity constraint: (1) active, (0) not active
param.max_iter_B2=300;
param.tight_L1 = 1; % Indicate if Psit is a tight frame (1) or not (0)
param.nu_L1 = 1;
param.max_iter_L1 = 20;
param.rel_obj_L1 = 1e-2;
    
    
% Solve BPSA problem
    
sol1 = sopt_mltb_solve_BPDN(y, epsilon, A, At, Psi, Psit, param);
    
% SARA
% It uses the solution to BPSA as a warm start
maxiter=10;
sigma=sigma_noise*sqrt(numel(y)/(numel(im)*8));
tol=1e-3;
  
sol2 = sopt_mltb_solve_rwBPDN(y, epsilon, A, At, Psi, Psit, param, sigma, tol, maxiter, sol1);

RSNR2=20*log10(norm(im,'fro')/norm(im-sol2,'fro'));


% Solve BPSA problem
    
sol12 = sopt_mltb_solve_BPDN(y2, epsilon, A2, At2, Psi, Psit, param);
    
% SARA
% It uses the solution to BPSA as a warm start
maxiter=10;
sigma=sigma_noise*sqrt(numel(y2)/(numel(im)*8));
tol=1e-3;
  
sol22 = sopt_mltb_solve_rwBPDN(y2, epsilon, A2, At2, Psi, Psit, param, sigma, tol, maxiter, sol12);

RSNR22=20*log10(norm(im,'fro')/norm(im-sol22,'fro'));

% Solve BPSA problem
    
sol13 = sopt_mltb_solve_BPDN(y3, epsilon, A3, At3, Psi, Psit, param);

% SARA
% It uses the solution to BPSA as a warm start
maxiter=10;
sigma=sigma_noise*sqrt(numel(y3)/(numel(im)*8));
tol=1e-3;
  
sol23 = sopt_mltb_solve_rwBPDN(y3, epsilon, A3, At3, Psi, Psit, param, sigma, tol, maxiter, sol13);

RSNR23=20*log10(norm(im,'fro')/norm(im-sol23,'fro'));

% Solve BPSA problem
    
sol14 = sopt_mltb_solve_BPDN(y4, epsilon, A4, At4, Psi, Psit, param);

% SARA
% It uses the solution to BPSA as a warm start
maxiter=10;
sigma=sigma_noise*sqrt(numel(y4)/(numel(im)*8));
tol=1e-3;
  
sol24 = sopt_mltb_solve_rwBPDN(y4, epsilon, A4, At4, Psi, Psit, param, sigma, tol, maxiter, sol14);

RSNR24=20*log10(norm(im,'fro')/norm(im-sol24,'fro'));
%%
%Show reconstructed images

rsnr=[RSNR2,RSNR22,RSNR24,RSNR23];

save('fig1.mat','im','sol2','sol22','sol23','sol24','rsnr');

figure, imagesc(sol2,[0 1]); axis image; axis off; colormap gray;
title(['Random Convolution, SNR=',num2str(RSNR2), 'dB'])


figure, imagesc(sol22,[0 1]); axis image; axis off; colormap gray;
title(['Partial Random Circulant, SNR=',num2str(RSNR22), 'dB'])

figure, imagesc(sol23,[0 1]); axis image; axis off; colormap gray;
title(['Partial Random Circulant w/ Golay, SNR=',num2str(RSNR23), 'dB'])

figure, imagesc(sol24,[0 1]); axis image; axis off; colormap gray;
title(['Random Convolution w/ Golay, SNR=',num2str(RSNR24), 'dB'])









