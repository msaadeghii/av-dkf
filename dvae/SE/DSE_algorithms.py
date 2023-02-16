#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright (c) 2020 by Inria
Authoried by Xiaoyu BIE (xiaoyu.bie@inrai.fr)
License agreement in LICENSE.txt

This function can be used specifically in speech enhancement based on RVAE, DKF and SRNN-v1

"""

import os
import time
import torch
import numpy as np
import librosa
import soundfile as sf
from iter  import concat_iter
from abc import ABC, abstractmethod



class EM(ABC):

    def __init__(self, X, Vf, W, H, g, R, niter=100, device='cpu', boosting=False):

        self.device = device
        
        self.X = X # (F, N) niosy STFT
        self.X_abs_2 = np.abs(X) ** 2 # (F, N) mixture power spectrogram
        self.Vf = self.np2tensor(Vf).to(self.device)  # video frames
        self.Vf = self.Vf #.permute(2,0,1)
        self.W = W # (F, K) NMF noise model
        self.H = H # (K, N) NMF moise model
        self.g = g # ( , N) gain parameters, varies cross time
        self.niter = niter # number of iteration for EM steps

        
        
        F, N = self.X.shape
        self.Vs = torch.zeros(R, F, N).to(self.device) # (R, F, N) variance of clean speech
        self.Vs_scaled = torch.zeros_like(self.Vs).to(self.device) # (R, F, N) Vs multiplied by gain
        self.Vx = torch.zeros_like(self.Vs).to(self.device) # (R, F, N) variance of noisy speech

        self.params_np2tensor()
        self.compute_Vb()

    def np2tensor(self, x):
        y = torch.from_numpy(x.astype(np.float32))
        return y
    
    def params_np2tensor(self):

        self.X_abs_2 = torch.from_numpy(self.X_abs_2.astype(np.float32)).to(self.device)
        self.W = torch.from_numpy(self.W.astype(np.float32)).to(self.device)
        self.H = torch.from_numpy(self.H.astype(np.float32)).to(self.device)
        self.g = torch.from_numpy(self.g.astype(np.float32)).to(self.device)


    @ abstractmethod
    def compute_Vs(self, Z):
        pass

    
    def compute_Vs_scaled(self):
        self.Vs_scaled = self.g * self.Vs

    
    def compute_Vb(self):
        self.Vb = self.W @ self.H


    def compute_Vx(self):
        self.Vx = self.Vs_scaled + self.Vb


    def compute_loss(self):
        return torch.mean(torch.log(self.Vx) + self.X_abs_2 / self.Vx)


    @ abstractmethod
    def E_step(self):
        pass


    def M_step(self):
        # M-step aims to update W, H and g

        # update W
        num = (self.X_abs_2 * torch.sum(self.Vx ** -2, dim=0)) @ self.H.T # (F, K)
        den = torch.sum(self.Vx ** -1, dim=0) @ self.H.T # (F, K)
        self.W = self.W * torch.sqrt(num / den) # (F, K)

        # update variancec
        self.compute_Vb()
        self.compute_Vx()

        # update H
        num = self.W.T @ (self.X_abs_2 * torch.sum(self.Vx ** -2, dim=0)) # (K, N)
        den = self.W.T @ torch.sum(self.Vx ** -1, axis=0) # (K, N)
        self.H = self.H * torch.sqrt(num / den) # (K, N)

        # update variance of noise and noisy speech
        self.compute_Vb()
        self.compute_Vx()
    
        # normalize W and H, don't change Vb
        norm_col_W = torch.sum(torch.abs(self.W), dim=0)
        self.W = self.W / norm_col_W.unsqueeze(0)
        self.H = self.H * norm_col_W.unsqueeze(1)
        
        # update g
        num = torch.sum(self.X_abs_2 * torch.sum(self.Vs * (self.Vx ** -2), dim=0), dim=0) # (, N)
        den = torch.sum(torch.sum(self.Vs * (self.Vx ** -1), dim=0), dim=0) # (, N)
        self.g = self.g * torch.sqrt(num/den) # (, N)

        # update variance of scaled clean speech and noisy speech
        self.compute_Vs_scaled()
        self.compute_Vx()


    @ abstractmethod
    def compute_WienerFilter(self):
        pass


    def run(self, tqdm=None):

        loss = np.zeros(self.niter)
        time_consume = np.zeros(self.niter)

        start_time = time.time()

        for n in range(self.niter):

            time_cut = time.time()
            # update encoder
            self.E_step()

            # update noise params and gain
            self.M_step()

            loss[n] = self.compute_loss()
            time_consume[n] = time.time() - time_cut

            print('iter: {}/{} - loss: {:.4f} - time cost: {:.1f}s'.format(n+1, self.niter, loss[n], time_consume[n]))

        end_time = time.time()
        elapsed = (end_time - start_time) / 60
        print('elapsed time: {:.2f} min'.format(elapsed))

        WFs, WFn = self.compute_WienerFilter()

        self.S_hat = WFs * self.X
        self.N_hat = WFn * self.X

        return loss, time_consume
    

    def run_monitor_EM(self, fs, hop, wlen, win, audio_ref):

        tmp_recon = '/local_scratch/xbie/Results/tmp/tmp_recon.wav'
        eval_metrics = EvalMetrics(metric='all')

        loss = np.zeros(self.niter)
        rmse = np.zeros(self.niter)
        pesq = np.zeros(self.niter)
        estoi = np.zeros(self.niter)

        start_time = time.time()
        print('EM start')

        for n in range(self.niter):

            # update encoder
            self.E_step()

            # update noise params and gain
            self.M_step()

            # calculate loss
            loss[n] = self.compute_loss()

            # calculate Wiener filter and reconstruct signal
            
            WFs, WFn = self.compute_WienerFilter()
            self.S_hat = WFs * self.X
            s_hat = librosa.istft(stft_matrix=self.S_hat, hop_length=hop,
                                  win_length=wlen, window=win)
            sf.write(tmp_recon, s_hat, fs)

            # evaluation
            rmse[n], pesq[n], estoi[n] =  eval_metrics.eval(audio_est=tmp_recon, audio_ref=audio_ref)

            print('iter: {}/{} - loss: {:.4f} - rmse: {:.4f} - pesq: {:.4f} - estoi: {:.4f}'.format(n, self.niter-1, loss[n], rmse[n], pesq[n], estoi[n]))

        # delete tmp audio file
        os.remove(tmp_recon)

        # calculate total computation time
        end_time = time.time()
        elapsed = (end_time - start_time) / 60
        print('elapsed time: {:.2f} min'.format(elapsed))

        WFs, WFn = self.compute_WienerFilter()

        self.S_hat = WFs * self.X
        self.N_hat = WFn * self.X

        return loss, rmse, pesq, estoi

            

class DVEM(EM):

    def __init__(self, X, Vf, W, H, g, dvae, dvae_type, niter=100, 
                 nepochs_E_step=10, lr=1e-2, nsamples_E_step=1,  
                 nsamples_WF=1, device='cpu'):

        super().__init__(X=X, Vf=Vf, W=W, H=H, g=g, R=nsamples_E_step, niter=niter, device=device)

        # E-step params
        self.nepochs_E_step =nepochs_E_step
        self.lr = lr
        self.nsamples_E_step = nsamples_E_step
        self.nsamples_WF = nsamples_WF

        # DVAE model params
        self.dvae = dvae
        self.dvae_type = dvae_type
        self.iter_params = []
        
        # Train inference part during E-step
        if self.dvae_type == 'DKF':
            self.iter_params = concat_iter(self.dvae.mlp_x_gx.parameters(),
                                           self.dvae.rnn_gx.parameters(),
                                           self.dvae.mlp_ztm1_g.parameters(),
                                           self.dvae.mlp_g_z.parameters(),
                                           self.dvae.inf_mean.parameters(),
                                           self.dvae.inf_logvar.parameters())
        elif self.dvae_type == 'RVAE':
            self.iter_params = concat_iter(self.dvae.mlp_x_gx.parameters(),
                                           self.dvae.rnn_g_x.parameters(),
                                           self.dvae.mlp_z_gz.parameters(),
                                           self.dvae.rnn_g_z.parameters(),
                                           self.dvae.mlp_g_z.parameters(),
                                           self.dvae.inf_mean.parameters(),
                                           self.dvae.inf_logvar.parameters())
        elif self.dvae_type == 'SRNN-var-detach-progressive':
            self.iter_params = concat_iter(self.dvae.mlp_hx_g.parameters(),
                                           self.dvae.rnn_g.parameters(),
                                           self.dvae.mlp_gz_z.parameters(),
                                           self.dvae.inf_mean.parameters(),
                                           self.dvae.inf_logvar.parameters())
        else:
            raise NameError('Unexpected DVAE type')

        self.optimizer = torch.optim.Adam(self.iter_params, lr=self.lr)

        # Set in training mode
        self.dvae.train()


    def sample_Z(self, nsamples=5):
        
        # (F, N) -> (1, F, N) -> (N, 1, F) -> (N, R, F)
        data_batch = self.X_abs_2.unsqueeze(0).permute(-1, 0, 1).repeat(1, nsamples, 1)
        data_batch_v = self.Vf.unsqueeze(0).permute(0,-1,1)
        with torch.no_grad():
            Z, _, _ = self.dvae.inference(data_batch, data_batch_v) # (N, R, F) -> (N, R, L), R is considered as batch size
        
        return Z


    def compute_Vs(self, Z):
        
        with torch.no_grad():
            data_batch_v = self.Vf.unsqueeze(0).permute(0,-1,1)
            Vs = self.dvae.generation_x(Z, data_batch_v) # (N, R, L) -> (N, R, F)

        self.Vs = Vs.permute(1, -1, 0) # (R, F, N)


    def E_step(self):

        for epoch in range(self.nepochs_E_step):

            self.optimizer.zero_grad()

            Vs = self.dvae(self.X_abs_2, self.Vf) # (F, N), dvae will adapt dimension automatically
            Vx = self.g * Vs + self.Vb # (F, N)
            z_mean = self.dvae.z_mean
            z_logvar = self.dvae.z_logvar

            if self.dvae_type == 'DKF' or self.dvae_type == 'SRNN-var-detach-progressive':
                z_mean_p = self.dvae.z_mean_p
                z_logvar_p = self.dvae.z_logvar_p
                
                loss_recon = torch.sum(self.X_abs_2 / Vx + torch.log(Vx))
                loss_KLD = - 0.5 * torch.sum(z_logvar - z_logvar_p 
                                - torch.div(z_logvar.exp() + (z_mean - z_mean_p).pow(2), z_logvar_p.exp()))
                loss_tot = loss_recon + loss_KLD
            elif self.dvae_type == 'RVAE':
                loss_recon = torch.sum(self.X_abs_2 / Vx + torch.log(Vx))
                loss_KLD = - 0.5 * torch.sum(z_logvar - z_logvar.exp() - z_mean.pow(2))
                loss_tot = loss_recon + loss_KLD

            loss_tot.backward()

            self.optimizer.step()

        # sample Z
        Z = self.sample_Z(self.nsamples_E_step) # (N, R, L)

        # compute variance
        self.compute_Vs(Z)
        self.compute_Vs_scaled()
        self.compute_Vx()

    
    def compute_WienerFilter(self):

        # sample z
        Z = self.sample_Z(self.nsamples_WF) # (N, R, L)
        
        # compute variance
        self.compute_Vs(Z) # (R, F, N)
        self.compute_Vs_scaled()
        self.compute_Vx()
        
        # compute Wiener Filters
        WFs = torch.mean(self.Vs_scaled / self.Vx, dim=0) # (F, N)
        WFn = torch.mean(self.Vb / self.Vx, dim=0) # (, N)

        WFs = WFs.cpu().detach().numpy()
        WFn = WFn.cpu().detach().numpy()

        return WFs, WFn


"""
Specifically for SRNN
In this algorithm, rather than fin-tuning the encoder network, we choose to update s_t 
iteratively using probabilistic Winer filter

Since we need the generation of z_t to approximate the latent variable, so we need 
"""
class DVEM_fast(EM):

    def __init__(self, X, W, H, g, dvae, dvae_type, niter=100, lr=1e-2, nsamples_E_step=1,  
                 nsamples_WF=1, device='cpu'):

        super().__init__(X=X, W=W, H=H, g=g, R=1, niter=niter, device=device)

        # E-step params
        # self.nepochs_E_step =nepochs_E_step
        self.lr = lr
        self.nsamples_E_step = nsamples_E_step
        self.nsamples_WF = nsamples_WF

        # DVAE model params
        self.dvae = dvae
        self.dvae_type = dvae_type
        self.device = device
        
        # Wienner filter
        self.WFs = np.zeros_like(X)
        self.WFn = np.zeros_like(X)


    def E_step(self):

        x_dim, seq_len = self.X.shape
        z_dim = self.dvae.z_dim
        bs = self.nsamples_E_step # batch size
        z_tm1 = torch.zeros(bs, z_dim).to(self.device) # z_0
        s_tm1 = torch.zeros(1, bs, x_dim).to(self.device) # s_0

        h_h_t = torch.zeros(self.dvae.num_RNN_h, bs, self.dvae.dim_RNN_h).to(self.device)
        c_h_t = torch.zeros(self.dvae.num_RNN_h, bs, self.dvae.dim_RNN_h).to(self.device)

        for t in range(seq_len):

            # deterministic_h
            s_h = self.dvae.mlp_x_h(s_tm1)
            _, (h_h_t, c_h_t) = self.dvae.rnn_h(s_h, (h_h_t, c_h_t))
            h_h_t_last = h_h_t.view(self.dvae.num_RNN_g, 1, bs, self.dvae.dim_RNN_h)[-1, :,:,:] # last layer of h_h_t
            h_h_t_last = h_h_t_last.view(bs, self.dvae.dim_RNN_h) # (1,1,bs,dim_RNN_h) -> (bs,dim_RNN_h)

            # generation_z
            hz_z = torch.cat((h_h_t_last, z_tm1), -1)
            hz_z = self.dvae.mlp_hz_z(hz_z)
            z_mean_t = self.dvae.prior_mean(hz_z)
            z_logvar_t = self.dvae.prior_logvar(hz_z)
            z_t = self.dvae.reparameterization(z_mean_t, z_logvar_t)

            # generation_s
            hz_s = torch.cat((h_h_t_last, z_t), -1)
            hz_s = self.dvae.mlp_hz_x(hz_s)
            log_Vs_t = self.dvae.gen_logvar(hz_s)
            tilde_Vs_t = torch.exp(log_Vs_t) # (bs, x_dim)

            # compute Wienner Filters
            tilde_Vs_scaled_t = self.g[t] * tilde_Vs_t # (bs, x_dim)
            tilde_Vx_t = tilde_Vs_scaled_t + self.Vb[:,t] # (bs, x_dim)
            WFs_t = torch.mean(tilde_Vs_scaled_t/tilde_Vx_t)
            WFn_t = torch.mean(self.Vb[:,t]/tilde_Vx_t)

            self.WFs[:,t] = WFs_t.cpu().detach().numpy()
            self.WFn[:,t] = WFn_t.cpu().detach().numpy()

            # estimate s_t
            # s = WFs * X => |s|^2 = WFs^2 * X_abs_2
            Vs_t = (WFs_t ** 2)  * self.X_abs_2[:,t]
            self.Vs[0,:,t] = Vs_t

            # update z_t
            z_tm1 = z_t
            s_tm1 = Vs_t.view(1, 1, x_dim).repeat(1, bs, 1)


        self.compute_Vs_scaled()
        self.compute_Vx()

    
    def compute_WienerFilter(self):

        return self.WFs, self.WFn

    def compute_Vs(self):

        return self.Vs
