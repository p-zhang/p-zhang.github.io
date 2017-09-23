%% Experiment1
% In this experiment we evaluate the performance of SARA for spread 
% spectrum acquisition. We use a 256x256 version of Lena as a test image. 
% Number of measurements is M = 0.2N and input SNR is set to 30 dB. These
% parameters can be changed by modifying the variables p (for the
% undersampling ratio) and input_snr (for the input SNR).


%% Clear workspace

clc
close all
clear all
clear;


%% Define paths

addpath misc/
addpath prox_operators/
addpath test_images/



%% Read image

imagename1 = 'elaine.pgm';
imagename2 = 'astro1.pgm';
imagename3 = 'nat3.pgm';
imagename4 = 'indor1.pgm';

% Load image
im1 = im2double(imread(imagename1));
im2 = im2double(imread(imagename2));
im3 = im2double(imread(imagename3));
im4 = im2double(imread(imagename4));

% Normalise
im1 = im1/max(max(im1));
im2 = im2/max(max(im2));
im3 = im3/max(max(im3));
im4 = im4/max(max(im4));

% Enforce positivity
im1(im1<0) = 0;
im2(im2<0) = 0;
im3(im3<0) = 0;
im4(im4<0) = 0;
im=[im1,im2;im3,im4];

img=zeros(256,256,4);

img(:,:,1)=im1;
img(:,:,2)=im2;
img(:,:,3)=im3;
img(:,:,4)=im4;

%% Parameters

input_snr = 30; % Noise level (on the measurements)

%% Sparsity operators

%Wavelet decomposition depth
nlevel=4;
for i=1:4
dwtmode('per');
[C(i,:),S(:,:,i)]=wavedec2(img(:,:,i),nlevel,'db8'); 
ncoef(i)=length(C(i,:));
[C1(i,:),S1(:,:,i)]=wavedec2(img(:,:,i),nlevel,'db1'); 
ncoef1(i)=length(C1(i,:));
[C2(i,:),S2(:,:,i)]=wavedec2(img(:,:,i),nlevel,'db2'); 
ncoef2(i)=length(C2(i,:));
[C3(i,:),S3(:,:,i)]=wavedec2(img(:,:,i),nlevel,'db3'); 
ncoef3(i)=length(C3(i,:));
[C4(i,:),S4(:,:,i)]=wavedec2(img(:,:,i),nlevel,'db4'); 
ncoef4(i)=length(C4(i,:));
[C5(i,:),S5(:,:,i)]=wavedec2(img(:,:,i),nlevel,'db5'); 
ncoef5(i)=length(C5(i,:));
[C6(i,:),S6(:,:,i)]=wavedec2(img(:,:,i),nlevel,'db6'); 
ncoef6(i)=length(C6(i,:));
[C7(i,:),S7(:,:,i)]=wavedec2(img(:,:,i),nlevel,'db7'); 
ncoef7(i)=length(C7(i,:));
end
%SARA

Psit = @(x) [wavedec2(x,nlevel,'db1')'; wavedec2(x,nlevel,'db2')';wavedec2(x,nlevel,'db3')';...
    wavedec2(x,nlevel,'db4')'; wavedec2(x,nlevel,'db5')'; wavedec2(x,nlevel,'db6')';...
    wavedec2(x,nlevel,'db7')';wavedec2(x,nlevel,'db8')']/sqrt(8); 

Psi = @(x,i) (waverec2(x(1:ncoef1(i)),S1(:,:,i),'db1')+waverec2(x(ncoef1(i)+1:ncoef1(i)+ncoef2(i)),S2(:,:,i),'db2')+...
    waverec2(x(ncoef1(i)+ncoef2(i)+1:ncoef1(i)+ncoef2(i)+ncoef3(i)),S3(:,:,i),'db3')+...
    waverec2(x(ncoef1(i)+ncoef2(i)+ncoef3(i)+1:ncoef1(i)+ncoef2(i)+ncoef3(i)+ncoef4(i)),S4(:,:,i),'db4')+...
    waverec2(x(ncoef1(i)+ncoef2(i)+ncoef3(i)+ncoef4(i)+1:ncoef1(i)+ncoef2(i)+ncoef3(i)+ncoef4(i)+ncoef5(i)),S5(:,:,i),'db5')+...
    waverec2(x(ncoef1(i)+ncoef2(i)+ncoef3(i)+ncoef4(i)+ncoef5(i)+1:ncoef1(i)+ncoef2(i)+ncoef3(i)+ncoef4(i)+ncoef5(i)+ncoef6(i)),S6(:,:,i),'db6')+...
    waverec2(x(ncoef1(i)+ncoef2(i)+ncoef3(i)+ncoef4(i)+ncoef5(i)+ncoef6(i)+1:ncoef1(i)+ncoef2(i)+ncoef3(i)+ncoef4(i)+ncoef5(i)+ncoef6(i)+ncoef7(i)),S7(:,:,i),'db7')+...
    waverec2(x(ncoef1(i)+ncoef2(i)+ncoef3(i)+ncoef4(i)+ncoef5(i)+ncoef6(i)+ncoef7(i)+1:ncoef1(i)+ncoef2(i)+ncoef3(i)+ncoef4(i)+ncoef5(i)+ncoef6(i)+ncoef7(i)+ncoef(i)),S(:,:,i),'db8'))/sqrt(8);

Psix = @(x) [Psi(x(1:numel(x)/4),1),Psi(x(numel(x)/4+1:2*numel(x)/4),2);Psi(x(2*numel(x)/4+1:3*numel(x)/4),3),Psi(x(3*numel(x)/4+1:numel(x)),4)];
Psitx = @(x) [Psit(x(1:256,1:256));Psit(x(1:256,257:512));Psit(x(257:512,1:256));Psit(x(257:512,257:512))];

%% Spread spectrum operator
   
ss=rand(size(im));
D=(2*(ss<0.5)-1);


B=sparse(1:512^2/4,1:512^2/4,1);
Bm=repmat(B,4);
Ma=sparse(1:512^2/4,1:512^2/4,ones(512^2/4,1),512^2/4,512^2);

A = @(x) Ma*Bm*reshape(D.*x,numel(x),1);
At = @(x) D.*reshape(Bm*(Ma'*x(:))*0.25,size(im));


% Sampling
y = A(im);
figure, imagesc(reshape(y,256,256),[0 1]); axis image; axis off; colormap gray;
title(['compressed image'])

%Add Gaussian i.i.d. noise
sigma_noise = 10^(-input_snr/20)*std(im(:));
y = y + (randn(size(y)) + 1i*randn(size(y)))*sigma_noise/sqrt(2);
comsed=abs(y);    
    
% Tolerance on noise
epsilon = sqrt(numel(y)+2*sqrt(numel(y)))*sigma_noise;
epsilon_up = sqrt(numel(y)+2.1*sqrt(numel(y)))*sigma_noise;
    
    
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
    
    

% Solve
    
sol3 = sopt_mltb_solve_BPDN(y, epsilon, A, At, Psix, Psitx, param);

maxiter=10;
sigma=sigma_noise*sqrt(numel(y)/(numel(im)));
tol=1e-3;
  
sol4 = sopt_mltb_solve_rwBPDN(y, epsilon, A, At, Psix, Psitx, param, sigma, tol, maxiter, sol3);
      
RSNR4=20*log10(norm(im,'fro')/norm(im-sol4,'fro'));
Rsnr1=20*log10(norm(im(1:256,1:256),'fro')/norm(im(1:256,1:256)-sol4(1:256,1:256),'fro'));
Rsnr2=20*log10(norm(im(1:256,257:512),'fro')/norm(im(1:256,257:512)-sol4(1:256,257:512),'fro'));
Rsnr3=20*log10(norm(im(257:512,1:256),'fro')/norm(im(257:512,1:256)-sol4(257:512,1:256),'fro'));
Rsnr4=20*log10(norm(im(257:512,257:512),'fro')/norm(im(257:512,257:512)-sol4(257:512,257:512),'fro'));

%Show reconstructed images

figure, imagesc(im,[0 1]); axis image; axis off; colormap gray;
title(['Original Images'])

figure, imagesc(sol4,[0 1]); axis image; axis off; colormap gray;
title(['SNRtt=',num2str(RSNR4), 'dB','SNR1=',num2str(Rsnr1), 'dB','SNR2=',num2str(Rsnr2), 'dB','SNR3=',num2str(Rsnr3), 'dB','SNR4=',num2str(Rsnr4), 'dB'])

% figure, imagesc(sol7,[0 1]); axis image; axis off; colormap gray;
% title(['TV, SNR=',num2str(RSNR7), 'dB'])
% figure, imagesc(sol8,[0 1]); axis image; axis off; colormap gray;
% title(['RW-TV, SNR=',num2str(RSNR8), 'dB'])

figure,

positionVector1=[0,0.6,0.2,0.2];
positionVector2=[0.2,0.6,0.2,0.2];
positionVector3=[0.4,0.6,0.2,0.2];
positionVector4=[0.6,0.6,0.2,0.2];
positionVector5=[0.8,0.45,0.2,0.2];
positionVector6=[0,0.3,0.2,0.2];
positionVector7=[0.2,0.3,0.2,0.2];
positionVector8=[0.4,0.3,0.2,0.2];
positionVector9=[0.6,0.3,0.2,0.2];

subplot('Position',positionVector5);
imagesc(reshape(comsed,256,256),[0 1]); axis image; axis off; colormap gray;
title(['v'])

subplot('Position',positionVector1);
imagesc(im(1:256,1:256),[0 1]); axis image; axis off; colormap gray;
title(['i'])
subplot('Position',positionVector2);
imagesc(im(1:256,257:512),[0 1]); axis image; axis off; colormap gray;
title(['ii'])
subplot('Position',positionVector3);
imagesc(im(257:512,1:256),[0 1]); axis image; axis off; colormap gray;
title(['iii'])
subplot('Position',positionVector4);
imagesc(im(257:512,257:512),[0 1]); axis image; axis off; colormap gray;
title(['iv'])

subplot('Position',positionVector6);
imagesc(sol4(1:256,1:256),[0 1]); axis image; axis off; colormap gray;
title(['vi'])
subplot('Position',positionVector7);
imagesc(sol4(1:256,257:512),[0 1]); axis image; axis off; colormap gray;
title(['vii'])
subplot('Position',positionVector8);
imagesc(sol4(257:512,1:256),[0 1]); axis image; axis off; colormap gray;
title(['viii'])
subplot('Position',positionVector9);
imagesc(sol4(257:512,257:512),[0 1]); axis image; axis off; colormap gray;
title(['ix'])


