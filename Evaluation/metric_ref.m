% function output = metric(A)
function output = metric(A, ref)
%% References %% Mean-squared error (MSE)---err = immse(A, ref);
mse_err = immse(A, ref);
%% References %% Peak Signal-to-Noise Ratio (PSNR)---[peaksnr, snr] = psnr(A, ref);
[peaksnr, snr] = psnr(A, ref);
%% References %% Structural Similarity Index (SSIM)---[ssimval, ssimmap] = ssim(A,ref);
[ssimval, ssimmap] = ssim(A,ref);
%% References %% Patch-based Contrast Quality Index(PCQI)---[mpcqi,pcqi_map]=PCQI(ref,A);
%[mpcqi, pcqi_map]=PCQI(ref, A);
%% No References %% Blind Contrast Enhancement Assessment by Gradient Ratioing at Visible Edges
%[Contrast_e, Contrast_ns, Contrast_r] = evaluation_mi(input, A);
%% No References %% Blur Metric
%blur_A = blurMetric(A);
%% No References %% Entropy
%entropy_A = entropy(A);
%% No References %% UIQM
% [UIQM_norm, UICM, UISM, UIConM] = UIQM(A);
%% No References %% UCIQE
% [UCIQE_norm,chroma,contrast,saturation] = UCIQE(A);
%% No Referencees %% NIQE
% niqe_A = niqe(A);
% templateModel = load('modelparameters.mat');
% mu_prisparam = templateModel.mu_prisparam;
% cov_prisparam = templateModel.cov_prisparam;
% niqe_A = computequality(A,96,96,0,0,mu_prisparam,cov_prisparam);
% clear templateModel mu_prisparam cov_prisparam;
%% No Referencees %% defade
%defade_A = DEFADE_value(A); %image oversize
%% BRISQUE
%brisque_A = brisquescore(A);
%% CCF
%ccf_A = CCF(A);
%% CPBDM
%gray_A = rgb2gray(A);
%cpbdm_A =  CPBD_compute(gray_A);
%% FADE
%fade_A = FADE(A);
%% ILNIQE
% templateModel = load('templatemodel.mat');
% templateModel = templateModel.templateModel;
% mu_prisparam = templateModel{1};
% cov_prisparam = templateModel{2};
% meanOfSampleData = templateModel{3};
% principleVectors = templateModel{4};
% ilniqe_A = il_computequality(A,mu_prisparam,cov_prisparam,principleVectors,meanOfSampleData);
% clear templateModel;
%% LPC
%lpc_A = lpc_color(A);
%% EME
eme_A = AUEOEME(A);
%% SSEQ
%sseq_A = SSEQ(A);
%% Combine
% output = [peaksnr,ssimval,mse_err];
output = [mse_err,peaksnr, ssimval];
end
