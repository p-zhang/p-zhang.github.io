%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation results for the structured sensing model proposed in our paper
%        
%        'Modulated Unit-Norm Tight Frames for Compressed Sensing'
%
%      Section IV.A. (Block) CS with arbitrary/determinnistic subsampler
%                     
% To compare the performance between the proposed block CS model with
% existing block CS model.
%
% Measruement Matrix (\Phi):
% 1) Full iid Gaussian (benchmark) 
% 2) Block iid Gaussian 
% 3) Block arbitrary subsampling + Hadamard + rand_diagonal 
%
% Sparsifying Basis (\Psi): normalize Hadamard/Fourier matrix
%
% Recovery Algorithm: FPC_AS
% 
% We fix the dimension of the signal and the sensing model, and vary the
% sparsity level of the signal. 500 trials are run for each sparsify level.
% We assume exact recovery if the signal-to-noise ratio is greater
% than 50dB. 
%
% Frequency of exact recovery = number of trials achieve exact recovery/100
%
% Simulation takes about 58 minutes on a computer with:
% Processor: Intel(R) Core(TM) i7-3770 CPU @ 3.4 GHz 3.4 GHz
% Installed memory (RAM): 8.00 GB
% System type: 64-bit Operating System
% Softwar: MATLAB R2012a (7.14.0.739)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


close all;
clear all;

%% Parameter setup

% Number of trials: to save time, one can reduce the number of trials.
% trial_num=60 is good enough to show the trend. 
%trial_num=500 makes the plot more smooth.
trial_num=500;

% Length of the signal
n=256;

% Number of observations
m=64;

% Number of blocks
L=1;

% Sparsity level
k=2:2:36;

%% Define the ortho basis
type = 0;

switch(type)
    case 0
        % the Hadamard basis
        Psi = hadamard(n)/sqrt(n);
    case 1
        % the Fourier basis
        Psi = conj(dftmtx(n))/sqrt(n);
end

%% Types of sensing models

% 1) iid Gaussian
Mtx1 = sqrt(1/m)*(randn(n,n));
Phi1 = Mtx1(1:m,:);

% 2) Block iid Gaussian
Mtx2 = randn(n,n);
Mtx2 = Mtx2(1:m,:);
p=m/L;q=n/L;
Blk2 = ones(p,q);
BLK2 = kron(eye(L),Blk2);
Phi2 = sqrt(1/p)*Mtx2.*BLK2;

% 3) Block arbitrary subsampling + Hadamard + rand_diagonal
rand_sqn = 2*binornd(1,0.5,1,n)-1;
Mtx3 = hadamard(q)/sqrt(q);
p3 = (1:p); 
Blk3 = sqrt(q/p)*Mtx3(p3,:);
Phi3 = kron(eye(L),Blk3)*diag(rand_sqn);

A1 = Phi1*Psi;
A2 = Phi2*Psi;
A3 = Phi3*Psi;
switch(type)
    case 1
        % the Fourier basis
        A1 = [real(A1(1:m/2,:));imag(A1(1:m/2,:))];
        A2 = [real(A2(1:m/2,:));imag(A2(1:m/2,:))];
        A3 = [real(A3(1:m/2,:));imag(A3(1:m/2,:))];
end
%% Measurement and Recovery

tic
for j=1:length(k); 
    
message=sprintf('Sparsity No.=%d',k(j));
disp(message);

rcnt1=0;
rcnt2=0;
rcnt3=0;

for r=1:trial_num

% Define the x and alp
alp = [randn(k(j),1); zeros(n-k(j),1)];
Psi = conj(dftmtx(n))/sqrt(n);
%Psi = hadamard(n)/sqrt(n);
pp = randperm(n);
alp = alp(pp);
x = Psi*alp;

% Observation
y1 = A1*alp;
y2 = A2*alp;
y3 = A3*alp;
     
% Reconstruction using FPC-AS
mu=1e-10; 
M = []; 
opts.gtol = 1e-8;
opts.record=0;
[alpr1, Out1] = FPC_AS(n,A1,y1,mu,M,opts);
[alpr2, Out2] = FPC_AS(n,A2,y2,mu,M,opts);
[alpr3, Out3] = FPC_AS(n,A3,y3,mu,M,opts);

% Recovered signal in spatial domain
xr1=Psi*alpr1;
xr2=Psi*alpr2;
xr3=Psi*alpr3;

% Calculate the SNR
snr1(r)=10*log10(mean(abs(x).^2)/mean((abs(x)-abs(xr1)).^2));
snr2(r)=10*log10(mean(abs(x).^2)/mean((abs(x)-abs(xr2)).^2));
snr3(r)=10*log10(mean(abs(x).^2)/mean((abs(x)-abs(xr3)).^2));

if snr1(r)>50
    rcnt1=rcnt1+1;
end

if snr2(r)>50
    rcnt2=rcnt2+1;
end
 
if snr3(r)>50
    rcnt3=rcnt3+1;
end

end
rcprb1(j)=rcnt1/trial_num;
rcprb2(j)=rcnt2/trial_num;
rcprb3(j)=rcnt3/trial_num;
end
toc
%% Plot the result
figure;
plot(k,rcprb1,'-ks')
grid on;
hold on
plot(k,rcprb2,'-m*')
plot(k,rcprb3,'-b^')
xlabel('Sparsity level');
ylabel('Frequency of Exact Recovery');
legend('iid Gaussian','Block iid Gaussian','Block Hadamard')
title(sprintf('Simulation results for different sensing models with dimension %dx%d',m,n));
