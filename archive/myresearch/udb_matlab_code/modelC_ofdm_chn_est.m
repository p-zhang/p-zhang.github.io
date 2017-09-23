%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Simulation results for the structured sensing model proposed in our paper
%        
%        'Modulated Unit-Norm Tight Frames for Compressed Sensing'
%
%   Section IV.C. OFDM channel estimation with low speed ADC and low PAPR
%                     
% To compare the performance between the proposed and existing CS based
% channel estimation methods for OFDM
%
% Types of methods
% 1) random pilot + low speed ADC
% 2) Golay pilot + random subsampling
% 3) Golay pilot + chirp sequence + low speed ADC
%
% Recovery Algorithm: Subspace Pursuit
% 
% 1000 trials are run for each input SNR.
%
% Simulation takes about 35 seconds on a computer with:
% Processor: Intel(R) Core(TM) i7-3770 CPU @ 3.4 GHz 3.4 GHz
% Installed memory (RAM): 8.00 GB
% System type: 64-bit Operating System
% Softwar: MATLAB R2012a (7.14.0.739)
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


close all;
clear all;
%%
N = 1024; % signal length

% 64 complex coefficient meausrements
M = 64;

thesize=N; 


tic
% unimodular random phase sequence 
para= exp(2*pi*sqrt(-1)*rand(N,1));  
fft_mtx=fft(eye(N));  
ifft_mtx=ifft(eye(N));
Rad_mat=ifft_mtx*diag(para)*fft_mtx;
Rad_fix_comp=Rad_mat(1:M,:);
Rad_fix=[real(Rad_fix_comp);imag(Rad_fix_comp)];


% Golay+Random sampling

[a,b]=generate_golay(10);
p2=randperm(N);
Golay_mat=ifft_mtx*diag(a)*fft_mtx;
Golay_mat=Golay_mat(p2(1:M),:);
Gol_op=[real(Golay_mat);imag(Golay_mat)];


% Golay+chirp sequence+det sampling
[c,d]=generate_golay(10);
p3=randperm(N);
rand_sqn1=2*binornd(1,0.5,1,N)-1;
Sigma=diag(rand_sqn1);
pone=[ones(1,N/M) zeros(1,N-N/M)];
P1=zeros(M,N);
for i=1:M
    P1(i,:)=circshift(pone',N*(i-1)/M)';
end
Det_Golay_mat=P1*Sigma*ifft_mtx*diag(c)*fft_mtx;
Det_Golay_op=[real(Det_Golay_mat);imag(Det_Golay_mat)];

x=zeros(N,1);
%% ATTC channel model
x(1)=1;
x(3)=0.31622;
x(18)=0.1995;
x(37)=0.1296;
x(76)=0.1;
x(138)=0.1;

% Noiseless simulation; 
% 30dB SNR;

SNR_db=0:2:30,


% Number of trials
trials_num=1000;

r=length(SNR_db);

SNR_Rec_rand=zeros(r,trials_num);
SNR_Golay_rand=zeros(r,trials_num);
SNR_Det_Golay_rand=zeros(r,trials_num);

for ii=1:trials_num

%SNR_Rec_golay=zeros(size(SNR_db));
%SNR_Rec_rand=SNR_Rec_golay;

y1 = Rad_fix*x;
sig_power1=sum(y1.*y1)/length(y1);


y2= Gol_op*x;
sig_power2=sum(y2.*y2)/length(y2);

y3= Det_Golay_op*x;
sig_power3=sum(y3.*y3)/length(y3);

for jj=1:length(SNR_db);
%jj
noise_vector=randn(size(y1));

noise_power1=sqrt(10^(-SNR_db(jj)/10)*sig_power1);
noise_y1=noise_power1*noise_vector;
y1_noise=y1+noise_y1;
%snr(y1,y1_noise)

noise_power2=sqrt(10^(-SNR_db(jj)/10)*sig_power2);
noise_y2=noise_power2*noise_vector;
y2_noise=y2+noise_y2;
%snr(y2,y2_noise)

noise_power3=sqrt(10^(-SNR_db(jj)/10)*sig_power3);
noise_y3=noise_power3*noise_vector;
y3_noise=y3+noise_y3;
%snr(y3,y3_noise)

 ep1 = noise_power1*sqrt(2*M)*sqrt(1 + 2*sqrt(2)/sqrt(2*M));
 ep2 = noise_power2*sqrt(2*M)*sqrt(1 + 2*sqrt(2)/sqrt(2*M)); 
 ep3 = noise_power3*sqrt(2*M)*sqrt(1 + 2*sqrt(2)/sqrt(2*M));
% 

alp1 = CSRec_SP(6, Rad_fix, y1_noise);
alp2= CSRec_SP(6,Gol_op,y2_noise);
alp3= CSRec_SP(6,Det_Golay_op,y3_noise);

SNR_Rec_rand(jj,ii)=snr(x,alp1);
SNR_Golay_rand(jj,ii)=snr(x,alp2);
SNR_Det_Golay_rand(jj,ii)=snr(x,alp3);
end
end

for kk=1:length(SNR_db)
aver_SNR_Rec_rand(kk)=sum(SNR_Rec_rand(kk,:))/trials_num;
aver_SNR_Golay_rand(kk)=sum(SNR_Golay_rand(kk,:))/trials_num;
aver_SNR_Det_Golay_rand(kk)=sum(SNR_Det_Golay_rand(kk,:))/trials_num;
end
toc
%%
plot(SNR_db,aver_SNR_Rec_rand,'b-*',SNR_db,aver_SNR_Golay_rand,'rx-.',SNR_db,aver_SNR_Det_Golay_rand,'go-.');
grid on
legend('Random Phase','Golay+Rand subsampling','Golay+Det subsampling','Location','northwest');
xlabel('Input SNR');
ylabel('Average Reconstructed SNR');
