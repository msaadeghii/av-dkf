#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code implements the speech enhancement method published in: S. Leglaive,
X. Alameda-Pineda, L. Girin, R. Horaud, "A recurrent variational autoencoder
for speech enhancement", IEEE International Conference on Acoustics Speech and
Signal Processing (ICASSP), Barcelona, Spain, 2020.

Copyright (c) 2019-2020 by Inria and CentraleSupelec
Authored by Simon Leglaive (simon.leglaive@centralesupelec.fr)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License (version 3)
as published by the Free Software Foundation.

You should have received a copy of the GNU Affero General Public License
along with this program (LICENSE.txt). If not, see
<https://www.gnu.org/licenses/>.

Adapted and modified by Mostafa Sadeghi (mostafa.sadeghi@inria.fr) and Ali Golmakani (golmakani77@yahoo.com).

"""

import sys
sys.path.append("../utils")
sys.path.append("../model")
sys.path.append("../..")

my_seed = 0
import numpy as np
np.random.seed(my_seed)
import torch
torch.manual_seed(my_seed)
from torch import optim
import time
from torch.autograd.functional import jacobian
from torch import nn
import librosa
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import shutil

class istft_speech:

    def __init__(self, stft_param):

        self.hop = stft_param["hop"]
        self.wlen = stft_param["wlen"]
        self.win = stft_param["win"]
        self.len = stft_param["len"]

    def run(self, S_hat):

        s_hat = librosa.istft(stft_matrix=S_hat, hop_length=self.hop,
                      win_length=self.wlen, window=self.win, length=self.len)

        return s_hat

class EM:
    """
    Meant to be an abstract class that should not be instantiated but only
    inherited.
    """

    def __init__(self, X, Vf, W, H, g, vae, num_iter=100, device='cpu', verbose = False, fix_gain = False):

        self.device = device
        self.verbose = verbose
        self.Vf = self.np2tensor(Vf).to(self.device)  # video frames
        self.Vf = self.Vf #.permute(2,0,1)
        self.X = X # mixture STFT, shape (F,N)
        self.X_abs_2 = np.abs(X)**2 # mixture power spectrogram, shape (F, N)
        self.W = W # NMF dictionary matrix, shape (F, K)
        self.H = H # NMF activation matrix, shape (K, N)
        self.compute_Vb() # noise variance, shape (F, N)
        self.vae = vae # variational autoencoder
        self.num_iter = num_iter # number of iterations
        self.g = g # gain parameters, shape (, N)
        self.Vs = None # speech variance, shape (R, F, N), where R corresponds
        # to different draws of the latent variables fed as input to the vae
        # decoder
        self.Vs_scaled = None # speech variance multiplied by gain,
        # shape (R, F, N)
        self.Vx = None # mixture variance, shape (R, F, N)
        self.fix_gain = fix_gain

    def np2tensor(self, x):
        y = torch.from_numpy(x.astype(np.float32))
        return y

    def tensor2np(self, x):
        y = x.cpu().numpy()
        return y

    def compute_expected_neg_log_like(self):
        return np.mean(np.log(self.Vx) + self.X_abs_2/self.Vx)

    def compute_Vs(self, Z):
        pass

    def compute_Vs_scaled(self):
        self.Vs_scaled = self.g * self.Vs

    def compute_Vx(self):
        self.Vx = self.Vs_scaled + self.Vb#[:,None,:]
        # self.Vx = np.transpose(self.Vx, (1,0,-1))

    def compute_Vb(self):
        self.Vb = self.W @ self.H

    def E_step(self):
        # The E-step aims to generate latent variables in order to compute the
        # cost function required for the M-step. These samples are then also
        # used to update the model parameters at the M-step.
        pass

    def M_step(self):

        # The M-step aims to update W, H and g
        if self.Vx.ndim == 2:
            # Vx and Vs are expected to be of shape (R, F, N) so we add a
            # singleton dimension.
            # This will happen for the PEEM algorithm only where there is no
            # sampling of the latent variables according to their posterior.
            rem_dim = True
            self.Vx = self.Vx[np.newaxis,:,:]
            self.Vs = self.Vs[np.newaxis,:,:]
            self.Vs_scaled = self.Vs_scaled[np.newaxis,:,:]
        else:
            rem_dim = False

        # update W
        num = (self.X_abs_2*np.sum(self.Vx**-2, axis=0)) @ self.H.T
        den = np.sum(self.Vx**-1, axis=0) @ self.H.T
        self.W = self.W*(num/den)**.5

        # update variances
        self.compute_Vb()
        self.compute_Vx()

        # update H
        num = self.W.T @ (self.X_abs_2*np.sum(self.Vx**-2, axis=0))
        den = self.W.T @ np.sum(self.Vx**-1, axis=0)

        self.H = self.H*(num/den)**.5

        # update variances
        self.compute_Vb()
        self.compute_Vx()

        # normalize W and H
        norm_col_W = np.sum(np.abs(self.W), axis=0)

        self.W = self.W/norm_col_W[np.newaxis,:]
        self.H = self.H*norm_col_W[:,np.newaxis]

        # Update g
        if not self.fix_gain:
            num = np.sum(self.X_abs_2*np.sum(self.Vs*(self.Vx**-2), axis=0),
                         axis=0)
            den = np.sum(np.sum(self.Vs*(self.Vx**-1), axis=0), axis=0)
            self.g = self.g*(num/den)**.5

        # remove singleton dimension if necessary
        if rem_dim:
            self.Vx = np.squeeze(self.Vx)
            self.Vs = np.squeeze(self.Vs)
            self.Vs_scaled = np.squeeze(self.Vs_scaled)

        # update variances
        self.compute_Vs_scaled()
        self.compute_Vx()

    def run(self, params={}):

        tqdm = params["tqdm"] if "tqdm" in params else None
        stft_param = params["stft_param"] if "stft_param" in params else None
        eval_metric = params["eval_metric"] if "eval_metric" in params else None
        s_orig = params["s_orig"] if "s_orig" in params else None
        spec_param = params["spec_param"] if "spec_param" in params else None
        sample_rate = params["sample_rate"] if "sample_rate" in params else None
        cost = np.zeros(self.num_iter)
        if spec_param is not None:
            if os.path.exists("temp"):
                shutil.rmtree("temp")
            os.makedirs("temp")
        t0 = time.time()
        list_cost = []
        list_score = []
        if tqdm is not None:
            pbar = tqdm(np.arange(self.num_iter))
        else:
            pbar = np.arange(self.num_iter)
        for n in pbar:

            self.E_step()

            self.M_step()

            cost[n] = self.compute_expected_neg_log_like() # this cost only
            # corresponds to the expectation of the negative log likelihood
            # taken w.r.t the posterior distribution of the latent variables.
            # It basically tells us if the model fits the observations.
            # We could also compute the full variational free energy.
            list_cost.append(cost[n])

            if stft_param is not None:
                WFs, WFn = self.compute_WF(sample=True)
                this_S_hat = WFs*self.X
                this_N_hat = WFn*self.X
                s_hat = librosa.istft(stft_matrix=this_S_hat, hop_length=stft_param["hop"],
                              win_length=stft_param["wlen"], window=stft_param["win"], length=stft_param["len"])
                score_sisdr, score_pesq, score_stoi = eval_metric.eval(s_hat, s_orig)
                list_score.append([score_sisdr, score_pesq, score_stoi])
                if spec_param is not None:

                    D1 = librosa.amplitude_to_db(np.abs(spec_param["mix"]), ref=np.max)
                    D2 = librosa.amplitude_to_db(np.abs(spec_param["clean"]), ref=np.max)
                    D3 = librosa.amplitude_to_db(np.abs(this_N_hat), ref=np.max)
                    D4 = librosa.amplitude_to_db(np.abs(this_S_hat), ref=np.max)
                    spec_figure, ((ax1, ax2, ax5, ax6), (ax3, ax4, ax7, ax8)) = plt.subplots(nrows=2, ncols=4)

                    img1 = librosa.display.specshow(D1, y_axis='log', x_axis='time',

                                                   sr=sample_rate, ax= ax1)
                    img2 = librosa.display.specshow(D2, y_axis='log', x_axis='time',

                                                   sr=sample_rate, ax= ax2)
                    img3 = librosa.display.specshow(D3, y_axis='log', x_axis='time',

                                                   sr=sample_rate, ax= ax3)
                    img4 = librosa.display.specshow(D4, y_axis='log', x_axis='time',

                                                   sr=sample_rate, ax= ax4)

                    ax5.plot([x[0] for x in list_score], label = "SDR", color="r")
                    ax6.plot([x[1] for x in list_score], label = "PESQ", color="g")
                    ax7.plot([x[2] for x in list_score], label = "STOI", color="b")
                    ax8.plot(list_cost, label="Cost")

                    spec_figure.savefig('temp/frame' + '%03d'%(n) + '.png')
                    plt.close(spec_figure)

                if tqdm is not None:
                    pbar.set_postfix({'sdr': score_sisdr, 'cost': cost[n]})
            else:
                if tqdm is not None:
                    pbar.set_postfix({'cost': cost[n]})

        elapsed = time.time() - t0
        
        WFs, WFn = self.compute_WF(sample=True)

        if WFs.ndim == 2:
            self.S_hat = WFs*self.X
            self.N_hat = WFn*self.X
        else:
            self.S_hat = np.mean(WFs, axis = 0) *self.X
            self.N_hat = np.mean(WFn, axis = 0) *self.X 

        if self.verbose:
            print("elapsed time: %.4f s" % (elapsed))

        return cost

class DEM:
    """
    Meant to be an abstract class that should not be instantiated but only
    inherited.
    """

    def __init__(self, X, Vf, W, H, g, vae, num_iter=100, device='cpu', verbose = False, fix_gain = False):

        self.device = device
        self.verbose = verbose
        self.Vf = self.np2tensor(Vf).to(self.device)  # video frames
        self.Vf = self.Vf #.permute(2,0,1)
        self.X = X # mixture STFT, shape (F,N)
        self.X_abs_2 = np.abs(X)**2 # mixture power spectrogram, shape (F, N)
        self.W = W # NMF dictionary matrix, shape (F, K)
        self.H = H # NMF activation matrix, shape (K, N)
        self.compute_Vb() # noise variance, shape (F, N)
        self.vae = vae # variational autoencoder
        self.num_iter = num_iter # number of iterations
        self.g = g # gain parameters, shape (, N)
        self.Vs = None # speech variance, shape (R, F, N), where R corresponds
        # to different draws of the latent variables fed as input to the vae
        # decoder
        self.Vs_scaled = None # speech variance multiplied by gain,
        # shape (R, F, N)
        self.Vx = None # mixture variance, shape (R, F, N)
        self.fix_gain = fix_gain

    def np2tensor(self, x):
        y = torch.from_numpy(x.astype(np.float32))
        return y

    def npc2tensor(self, x):
        y = torch.from_numpy(x)
        return y

    def tensor2np(self, x):
        y = x.cpu().numpy()
        return y

    def compute_expected_neg_log_like(self):
        return np.mean(np.log(self.Vx) + self.X_abs_2/self.Vx)

    def compute_Vs(self, Z):
        pass

    def compute_Vs_scaled(self):
        self.Vs_scaled = self.g * self.Vs

    def compute_Vx(self):
        self.Vx = self.Vs_scaled + self.Vb[:,None,:]
        self.Vx = np.transpose(self.Vx, (1,0,-1))

    def compute_Vb(self):
        self.Vb = self.W @ self.H

    def E_step(self):
        # The E-step aims to generate latent variables in order to compute the
        # cost function required for the M-step. These samples are then also
        # used to update the model parameters at the M-step.
        pass

    def M_step(self):
        # The M-step aims to update W, H and g
        if self.Vx.ndim == 2:
            # Vx and Vs are expected to be of shape (R, F, N) so we add a
            # singleton dimension.
            # This will happen for the PEEM algorithm only where there is no
            # sampling of the latent variables according to their posterior.
            rem_dim = True
            self.Vx = self.Vx[np.newaxis,:,:]
            self.Vs = self.Vs[np.newaxis,:,:]
            self.Vs_scaled = self.Vs_scaled[np.newaxis,:,:]
        else:
            rem_dim = False

        # update W
        num = (self.X_abs_2*np.sum(self.Vx**-2, axis=0)) @ self.H.T
        den = np.sum(self.Vx**-1, axis=0) @ self.H.T
        self.W = self.W*(num/den)**.5

        # update variances
        self.compute_Vb()
        self.compute_Vx()

        # update H
        num = self.W.T @ (self.X_abs_2*np.sum(self.Vx**-2, axis=0))
        den = self.W.T @ np.sum(self.Vx**-1, axis=0)
        self.H = self.H*(num/den)**.5

        # update variances
        self.compute_Vb()
        self.compute_Vx()

        # normalize W and H
        norm_col_W = np.sum(np.abs(self.W), axis=0)
        self.W = self.W/norm_col_W[np.newaxis,:]
        self.H = self.H*norm_col_W[:,np.newaxis]

        # Update g
        if not self.fix_gain:
            num = np.sum(self.X_abs_2*np.sum(self.Vs*(self.Vx**-2), axis=0),
                         axis=0)
            den = np.sum(np.sum(self.Vs*(self.Vx**-1), axis=0), axis=0)
            self.g = self.g*(num/den)**.5

        # remove singleton dimension if necessary
        if rem_dim:
            self.Vx = np.squeeze(self.Vx)
            self.Vs = np.squeeze(self.Vs)
            self.Vs_scaled = np.squeeze(self.Vs_scaled)

        # update variances
        self.compute_Vs_scaled()
        self.compute_Vx()


    def run(self, params={}):
        tqdm = params["tqdm"] if "tqdm" in params else None
        stft_param = params["stft_param"] if "stft_param" in params else None
        eval_metric = params["eval_metric"] if "eval_metric" in params else None
        s_orig = params["s_orig"] if "s_orig" in params else None
        spec_param = params["spec_param"] if "spec_param" in params else None
        sample_rate = params["sample_rate"] if "sample_rate" in params else None
        cost = np.zeros(self.num_iter)
        if spec_param is not None:
            if os.path.exists("temp"):
                shutil.rmtree("temp")
            os.makedirs("temp")


        t0 = time.time()
        list_cost = []
        list_score = []
        if tqdm is not None:
            pbar = tqdm(np.arange(self.num_iter))
        else:
            pbar = np.arange(self.num_iter)
        for n in pbar:


            self.E_step()

            self.M_step()

            cost[n] = self.compute_expected_neg_log_like() # this cost only
            # corresponds to the expectation of the negative log likelihood
            # taken w.r.t the posterior distribution of the latent variables.
            # It basically tells us if the model fits the observations.
            # We could also compute the full variational free energy.
            list_cost.append(cost[n])
            if stft_param is not None:
                this_WFs, this_WFn = self.compute_WF(sample=True)
                this_S_hat = this_WFs*self.X
                this_N_hat = this_WFn*self.X
                s_hat = librosa.istft(stft_matrix=this_S_hat, hop_length=stft_param["hop"],
                              win_length=stft_param["wlen"], window=stft_param["win"], length=stft_param["len"])
                score_sisdr, score_pesq, score_stoi = eval_metric.eval(s_hat, s_orig)
                list_score.append([score_sisdr, score_pesq, score_stoi])
                if spec_param is not None:


                    D1 = librosa.amplitude_to_db(np.abs(spec_param["mix"]), ref=np.max)
                    D2 = librosa.amplitude_to_db(np.abs(spec_param["clean"]), ref=np.max)
                    D3 = librosa.amplitude_to_db(np.abs(this_N_hat), ref=np.max)
                    D4 = librosa.amplitude_to_db(np.abs(this_S_hat), ref=np.max)
                    spec_figure, ((ax1, ax2, ax5, ax6), (ax3, ax4, ax7, ax8)) = plt.subplots(nrows=2, ncols=4)

                    img1 = librosa.display.specshow(D1, y_axis='log', x_axis='time',

                                                   sr=sample_rate, ax= ax1)
                    img2 = librosa.display.specshow(D2, y_axis='log', x_axis='time',

                                                   sr=sample_rate, ax= ax2)
                    img3 = librosa.display.specshow(D3, y_axis='log', x_axis='time',

                                                   sr=sample_rate, ax= ax3)
                    img4 = librosa.display.specshow(D4, y_axis='log', x_axis='time',

                                                   sr=sample_rate, ax= ax4)

                    ax5.plot([x[0] for x in list_score], label = "SDR", color="r")
                    ax6.plot([x[1] for x in list_score], label = "PESQ", color="g")
                    ax7.plot([x[2] for x in list_score], label = "STOI", color="b")
                    ax8.plot(list_cost, label="Cost")

                    spec_figure.savefig('temp/frame' + '%03d'%(n) + '.png')
                    plt.close(spec_figure)

                if tqdm is not None:
                    pbar.set_postfix({'sdr': score_sisdr, 'cost': cost[n]})
            else:
                if tqdm is not None:
                    pbar.set_postfix({'cost': cost[n]})

        elapsed = time.time() - t0

        WFs, WFn = self.compute_WF(sample=True)
        self.S_hat = WFs*self.X
        self.N_hat = WFn*self.X
        if self.verbose:
            print("elapsed time: %.4f s" % (elapsed))

        return cost


class PEEM(EM):

    def __init__(self, X, Vf, W, H, g, Z, vae, num_iter, device, lr=1e-2,
                 num_E_step=10, verbose = False):

        super().__init__(X=X, Vf = Vf, W=W, H=H, g=g, vae=vae, num_iter=num_iter,
             device=device)

        self.verbose = verbose
        # mixture power spectrogram as tensor, shape (F, N)
        self.X_abs_2_t = self.np2tensor(self.X_abs_2).to(self.device)

        # intial value for the latent variables
        self.Z = Z # shape (L, N)
        self.Z_t = self.np2tensor(self.Z).to(self.device)

        # optimizer for the E-step
        self.optimizer = optim.Adam([self.Z_t.requires_grad_()], lr=lr)

        # VERY IMPORTANT PARAMETERS: If 1, bad results in the FFNN VAE case
        self.num_E_step = num_E_step

        self.vae.eval() # vae in eval mode

        self.zmean = torch.tensor([0.0]).to(self.device)
        self.zlogvar = torch.tensor([0.0]).to(self.device)

    def compute_Vs(self, Z):
        """ Z: tensor of shape (N, L) """

        Vs_t= self.vae.generation_x(torch.t(Z), self.Vf) # (F, N)
        Vs_t = Vs_t.t()
        self.Vs = self.tensor2np(Vs_t.detach())

        return Vs_t

    def E_step(self):
        """ """
        # vector of gain parameters as tensor, shape (, N)
        g_t = self.np2tensor(self.g).to(self.device)

        # noise variances as tensor, shape (F, N)
        Vb_t = self.np2tensor(self.Vb).to(self.device)

        def closure():
            # reset gradients
            self.optimizer.zero_grad()
            # compute speech variance
            Vs_t = self.compute_Vs(self.Z_t)
            # compute likelihood variance
            Vx_t = g_t*Vs_t + Vb_t
            # compute loss, do backward and update latent variables
            loss = ( torch.sum(self.X_abs_2_t/Vx_t + torch.log(Vx_t)) +
                    torch.sum((self.Z_t-self.zmean).pow(2)/self.zlogvar.exp()) )
            loss.backward()
            return loss

        # Some optimization algorithms such as Conjugate Gradient and LBFGS
        # need to reevaluate the function multiple times, so we have to pass
        # in a closure that allows them to recompute our model. The closure
        # should clear the gradients, compute the loss, and return it.
        for epoch in np.arange(self.num_E_step):
            self.optimizer.step(closure)

        # update numpy array from the new tensor
        self.Z = self.tensor2np(self.Z_t.detach())

        # compute variances
        self.compute_Vs(self.Z_t)
        self.compute_Vs_scaled()
        self.compute_Vx()

    def compute_WF(self, sample=False):
        # sample parameter useless

        # compute Wiener Filters
        WFs = self.Vs_scaled/self.Vx
        WFn = self.Vb/self.Vx

        return WFs, WFn

# Implementation of the speech enhancement algorithm proposed in the following paper:
# [*] M. Sadeghi and R. Serizel, Fast and Efficient Speech Enhancement with Variational Autoencoders,
#     in IEEE International Conference on Acoustics Speech and Signal Processing (ICASSP), Rhodes island, June 2023. 
# The Langevin Dynamics (LD) part of the code is adapted from https://www.lyndonduong.com/sgmcmc/

class LDEM(EM):
    
    def __init__(self, X, Vf, W, H, g, Z, vae, num_iter, device, lr=1e-2, 
                 num_E_step=10, eta = 0.01, tv_param = 5, num_samples = 5, verbose = False):
        
        super().__init__(X=X, Vf = Vf, W=W, H=H, g=g, vae=vae, num_iter=num_iter, device=device)
                
        self.verbose = verbose
        self.eta = eta # noise variance for the proposal distribution
        self.num_samples = num_samples
        self.tv_param = tv_param
        # mixture power spectrogram as tensor, shape (F, N)
        self.X_abs_2_t = self.np2tensor(self.X_abs_2).to(self.device) 

        # intial value for the latent variables
        self.Z = Z 
        self.Z_t = self.np2tensor(self.Z[None,:,:] + np.sqrt(self.eta) * np.random.randn(self.num_samples, Z.shape[0], Z.shape[1]) ).to(self.device)
        self.lr = lr
        
        # optimizer for the E-step     
        self.optimizer = optim.SGD([self.Z_t.requires_grad_()], lr=1., momentum=0.)  # momentum is set to zero
        
        # VERY IMPORTANT PARAMETERS: If 1, bad results in the FFNN VAE case
        self.num_E_step = num_E_step 
        
        self.vae.eval() # vae in eval mode
         
        self.zmean, self.zlogvar, _ = self.vae.zprior(self.Vf)
        self.zmean, self.zlogvar = self.zmean.detach().t().to(self.device), self.zlogvar.detach().t().to(self.device) 
                
    def _noise(self, params): 
        """We are adding param+noise to each param."""
        std = np.sqrt(2 * self.lr)
        loss = 0.
        for param in params:
            noise = torch.randn_like(param) * std
            loss += (noise * param).sum()
        return loss
    
    def compute_Vs(self, Z):
        """ Z: tensor of shape (N, L) """

        v_ = self.Vf[None,:,:].repeat(Z.shape[0],1,1)
        Vs_t= self.vae.generation_x(torch.transpose(Z, 1, 2), v_) # (F, N)
        Vs_t = torch.transpose(Vs_t, 1, 2)
        self.Vs = self.tensor2np(Vs_t.detach())
        
        return Vs_t

    def total_variation_loss(self, Z):      
        tv = (torch.abs(Z[:,:,1:] - Z[:,:,:-1]).pow(1)).sum()
        return tv
    
    def E_step(self):
        """ """
        # vector of gain parameters as tensor, shape (, N)
        g_t = self.np2tensor(self.g).to(self.device)[None, :]
        
        # noise variances as tensor, shape (F, N)
        Vb_t = self.np2tensor(self.Vb).to(self.device)[None, :, :]
        
        def LD():
            # reset gradients
            self.optimizer.zero_grad()
            # compute speech variance
            Vs_t = self.compute_Vs(self.Z_t)
            # compute likelihood variance
            Vx_t = g_t*Vs_t + Vb_t
            # compute loss, do backward and update latent variables
            loss = ( torch.sum(self.X_abs_2_t/Vx_t + torch.log(Vx_t)) + 
                    torch.sum((self.Z_t-self.zmean).pow(2)/self.zlogvar.exp()) + self.tv_param * self.total_variation_loss(self.Z_t) )  * self.lr

            loss += self._noise(self.Z_t)  # add noise*param before calling backward!
            loss.backward()
            self.optimizer.step()
            
            return loss
        
        for epoch in np.arange(self.num_E_step):
            LD()  

        self.Z_t.data += torch.sqrt(torch.tensor([self.eta]).to(self.device)) * torch.randn_like(self.Z_t) 
        
        # update numpy array from the new tensor
        self.Z = self.tensor2np(self.Z_t.detach())
        
        # compute variances
        self.compute_Vs(self.Z_t)
        self.compute_Vs_scaled()
        self.compute_Vx()
        
    def compute_WF(self, sample=False):
        # sample parameter useless
        
        # compute Wiener Filters
        WFs = self.Vs_scaled/self.Vx
        WFn = self.Vb/self.Vx
        
        return WFs, WFn

    
#%% Dynamical PEEM, i.e., the PEEM for the DKF generative model.

class DPEEM(DEM):

    def __init__(self, X, Vf, W, H, g, Z, vae, num_iter, device, lr=1e-3,
                 num_E_step=1, fix_gain = True, verbose = False):

        super().__init__(X=X, Vf = Vf, W=W, H=H, g=g, vae=vae, num_iter=num_iter, fix_gain = True,
             device=device)

        self.lr = lr
        self.verbose = verbose
        # mixture power spectrogram as tensor, shape (F, N)
        self.X_abs_2_t = self.np2tensor(self.X_abs_2)[:,None,:].to(self.device)

        # intial value for the latent variables
        self.Z = Z # shape (L, 1, N)
        self.Z_t = self.np2tensor(self.Z).to(self.device)
        self.z_dim = Z.shape[0]
        # optimizer for the E-step
        self.optimizer = optim.Adam([self.Z_t.requires_grad_()], lr=self.lr)

        # VERY IMPORTANT PARAMETERS: If 1, bad results in the FFNN VAE case
        self.num_E_step = num_E_step

        self.vae.eval() # vae in eval mode

    def compute_Vs(self, Z):
        """ Z: tensor of shape (N, L) """

        Vs_t= self.vae.generation_x(Z.permute(-1,1,0), self.Vf.unsqueeze(0).permute(-1, 0, 1)).permute(-1,1,0) # (F, 1 ,N)

        self.Vs = self.tensor2np(Vs_t.detach())

        return Vs_t


    def E_step(self):
        """ """
        # vector of gain parameters as tensor, shape (, N)
        g_t = self.np2tensor(self.g).to(self.device)[None,None,:]

        # noise variances as tensor, shape (F, 1, N)
        Vb_t = self.np2tensor(self.Vb).to(self.device)[:,None,:]


        def closure():
            # reset gradients
            self.optimizer.zero_grad()
            # compute speech variance
            Vs_t = self.compute_Vs(self.Z_t)
            # compute likelihood variance
            Vx_t = g_t*Vs_t + Vb_t
            # compute prior params
            z_0 = torch.zeros(self.z_dim,1,1).to(self.device)
            z_tm1 = torch.cat([z_0, self.Z_t[:,:,:-1]], -1) # (L,1,F)
            _, z_mean_p, z_logvar_p = self.vae.generation_z(z_tm1.permute(-1,1,0), self.Vf.unsqueeze(0).permute(-1, 0, 1))
            z_mean_p = z_mean_p.permute(-1,1,0)
            z_logvar_p = z_logvar_p.permute(-1,1,0)
            # compute loss, do backward and update latent variables

            loss = ( torch.sum(self.X_abs_2_t/Vx_t + torch.log(Vx_t)) +
                    torch.sum((self.Z_t-z_mean_p).pow(2)/z_logvar_p.exp()) )
            loss.backward()
            return loss

        # Some optimization algorithms such as Conjugate Gradient and LBFGS
        # need to reevaluate the function multiple times, so we have to pass
        # in a closure that allows them to recompute our model. The closure
        # should clear the gradients, compute the loss, and return it.
        for epoch in np.arange(self.num_E_step):
            self.optimizer.step(closure)

        # update numpy array from the new tensor
        self.Z = self.tensor2np(self.Z_t.detach())

        # compute variances
        self.compute_Vs(self.Z_t)
        self.compute_Vs_scaled()
        self.compute_Vx()

    def compute_WF(self, sample=False):

        # compute Wiener Filters
        self.Vx_transpose = np.transpose(self.Vx, (1,0,-1))
        WFs = (self.Vs_scaled/self.Vx_transpose).squeeze()
        WFn = (self.Vb[:,None,:]/self.Vx_transpose).squeeze()

        return WFs, WFn

#%% Langevin Dynamics Dynamical PEEM, i.e., the LD for the DKF generative model.

class LDDPEEM(DEM):

    def __init__(self, X, Vf, W, H, g, Z, vae, num_iter, device, lr=1e-3,
                 num_E_step=1, fix_gain = True, verbose = False):

        super().__init__(X=X, Vf = Vf, W=W, H=H, g=g, vae=vae, num_iter=num_iter, fix_gain = True,
             device=device)

        self.verbose = verbose
        self.lr = 1e-4
        # mixture power spectrogram as tensor, shape (F, N)
        self.X_abs_2_t = self.np2tensor(self.X_abs_2)[:,None,:].to(self.device)

        # intial value for the latent variables
        self.Z = Z # shape (L, 1, N)
        self.Z_t = self.np2tensor(self.Z).to(self.device)
        self.z_dim = Z.shape[0]
        # optimizer for the E-step
        self.optimizer = optim.SGD([self.Z_t.requires_grad_()], lr=1., momentum=0.)  # momentum is set to zero

        # VERY IMPORTANT PARAMETERS: If 1, bad results in the FFNN VAE case
        self.num_E_step = num_E_step

        self.vae.eval() # vae in eval mode
        
    def _noise(self, params): 
        """We are adding param+noise to each param."""
        std = np.sqrt(2 * self.lr)
        loss = 0.
        for param in params:
            noise = torch.randn_like(param) * std
            loss += (noise * param).sum()
        return loss
    
    def compute_Vs(self, Z):
        """ Z: tensor of shape (N, L) """

        Vs_t= self.vae.generation_x(Z.permute(-1,1,0), self.Vf.unsqueeze(0).permute(-1, 0, 1)).permute(-1,1,0) # (F, 1 ,N)

        self.Vs = self.tensor2np(Vs_t.detach())

        return Vs_t


    def E_step(self):
        """ """
        # vector of gain parameters as tensor, shape (, N)
        g_t = self.np2tensor(self.g).to(self.device)[None,None,:]

        # noise variances as tensor, shape (F, 1, N)
        Vb_t = self.np2tensor(self.Vb).to(self.device)[:,None,:]


        def LD():
            # reset gradients
            self.optimizer.zero_grad()
            # compute speech variance
            Vs_t = self.compute_Vs(self.Z_t)
            # compute likelihood variance
            Vx_t = g_t*Vs_t + Vb_t
            # compute prior params
            z_0 = torch.zeros(self.z_dim,1,1).to(self.device)
            z_tm1 = torch.cat([z_0, self.Z_t[:,:,:-1]], -1) # (L,1,F)
            _, z_mean_p, z_logvar_p = self.vae.generation_z(z_tm1.permute(-1,1,0), self.Vf.unsqueeze(0).permute(-1, 0, 1))
            z_mean_p = z_mean_p.permute(-1,1,0)
            z_logvar_p = z_logvar_p.permute(-1,1,0)
            # compute loss, do backward and update latent variables

            loss = ( torch.sum(self.X_abs_2_t/Vx_t + torch.log(Vx_t)) +
                    torch.sum((self.Z_t-z_mean_p).pow(2)/z_logvar_p.exp()) )  * self.lr
            
            loss += self._noise(self.Z_t)  # add noise*param before calling backward!
            loss.backward()
            self.optimizer.step()
            
            return loss
        
        for epoch in np.arange(self.num_E_step):
            LD()
        
        # update numpy array from the new tensor
        self.Z = self.tensor2np(self.Z_t.detach())

        # compute variances
        self.compute_Vs(self.Z_t)
        self.compute_Vs_scaled()
        self.compute_Vx()

    def compute_WF(self, sample=False):

        # compute Wiener Filters
        self.Vx_transpose = np.transpose(self.Vx, (1,0,-1))
        WFs = (self.Vs_scaled/self.Vx_transpose).squeeze()
        WFn = (self.Vb[:,None,:]/self.Vx_transpose).squeeze()

        return WFs, WFn

    
class GPEEM(EM):

    def __init__(self, X, Vf, W, H, g, Z, vae, num_iter, device, lr=1e-2,
                 num_E_step=10, verbose = False):

        super().__init__(X=X, Vf = Vf, W=W, H=H, g=g, vae=vae, num_iter=num_iter,
             device=device, fix_gain = True)

        self.verbose = verbose
        self.a = 1.0
        self.b = 1.0
        # mixture power spectrogram as tensor, shape (F, N)
        self.X_abs_2_t = self.np2tensor(self.X_abs_2).to(self.device)

        # intial value for the latent variables
        self.g_t = self.np2tensor(self.g).to(self.device)
        self.Z = Z # shape (L, N)
        self.Z_t = self.np2tensor(self.Z).to(self.device)

        # optimizer for the E-step
        self.optimizer = optim.Adam([self.Z_t.requires_grad_()] + [self.g_t.requires_grad_()], lr=lr)

        # VERY IMPORTANT PARAMETERS: If 1, bad results in the FFNN VAE case
        self.num_E_step = num_E_step

        self.vae.eval() # vae in eval mode

        self.zmean = torch.tensor([0.0]).to(self.device)
        self.zlogvar = torch.tensor([0.0]).to(self.device)
        self.Z_old = self.Z_t

    def compute_Vs(self, Z):
        """ Z: tensor of shape (N, L) """

        Vs_t= self.vae.generation_x(torch.t(Z), self.Vf) # (F, N)
        Vs_t = Vs_t.t()
        self.Vs = self.tensor2np(Vs_t.detach())

        return Vs_t

    def E_step(self):
        """ """

        # noise variances as tensor, shape (F, N)
        Vb_t = self.np2tensor(self.Vb).to(self.device)

        def closure():
            # reset gradients
            self.optimizer.zero_grad()
            # compute speech variance
            Vs_t = self.compute_Vs(self.Z_t)
            # compute likelihood variance
            Vx_t = self.g_t*Vs_t + Vb_t
            # compute loss, do backward and update latent variables
            loss = ( torch.sum(self.X_abs_2_t/Vx_t + torch.log(Vx_t)) +
                    torch.sum((self.Z_t-self.zmean).pow(2)/self.zlogvar.exp()) -
                    torch.sum((self.a-1)*torch.log(self.g_t) - self.b*self.g_t) )
            loss.backward()
            return loss

        # Some optimization algorithms such as Conjugate Gradient and LBFGS
        # need to reevaluate the function multiple times, so we have to pass
        # in a closure that allows them to recompute our model. The closure
        # should clear the gradients, compute the loss, and return it.
        for epoch in np.arange(self.num_E_step):
            self.optimizer.step(closure)
            self.g_t.data = torch.clip(self.g_t.data, min=0.001)
            
        # update numpy array from the new tensor
        self.Z = self.tensor2np(self.Z_t.detach())

        # compute variances
        self.compute_Vs(self.Z_t)
        self.compute_Vs_scaled()
        self.compute_Vx()

        self.Z_old = self.Z_t.clone()
        self.g = self.tensor2np(self.g_t.detach())


    def compute_WF(self, sample=False):

        # compute Wiener Filters
        WFs = self.Vs_scaled/self.Vx
        WFn = self.Vb/self.Vx

        return WFs, WFn


class GDPEEM(DEM):

    def __init__(self, X, Vf, W, H, g, Z, vae, num_iter, device, lr=1e-3, visual = None, is_z_oracle = False, is_noise_oracle = False, fix_gain = False,
                 num_E_step=1, verbose = False, Z_oracle = None, rec_power = 0.99):

        super().__init__(X=X, Vf = Vf, W=W, H=H, g=g, vae=vae, num_iter=num_iter,
             device=device, fix_gain = True)

        self.verbose = verbose
        # mixture power spectrogram as tensor, shape (F, N)
        self.X_abs_2_t = self.np2tensor(self.X_abs_2)[:,None,:].to(self.device)

        # intial value for the latent variables
        self.g_t = self.np2tensor(self.g).to(self.device)[None,None,:]
        self.Z = Z # shape (L, 1, N)
        self.Z_t = self.np2tensor(self.Z).to(self.device)
        if Z_oracle is not None:
            self.Z_oracle = Z_oracle # shape (L, N)
            self.Z_oracle_t = self.np2tensor(self.Z_oracle).to(self.device)

        self.visual = visual

        self.z_dim = Z.shape[0]
        # optimizer for the E-step
        self.optimizer = optim.Adam([self.Z_t.requires_grad_()] + [self.g_t.requires_grad_()], lr=lr)

        # VERY IMPORTANT PARAMETERS: If 1, bad results in the FFNN VAE case
        self.num_E_step = num_E_step

        self.vae.eval() # vae in eval mode

        self.is_z_oracle = is_z_oracle
        self.is_noise_oracle = is_noise_oracle
        self.a = 1.1
        self.b = 1.1

        self.rec_power = rec_power

    def compute_Vs(self, Z):
        """ Z: tensor of shape (N, L) """

        Vs_t= self.vae.generation_x(Z.permute(-1,1,0), self.visual).permute(-1,1,0) # (F, 1 ,N)

        self.Vs_t = Vs_t.clone().detach()
        self.Vs = self.tensor2np(Vs_t.detach())

        return Vs_t


    def E_step(self):
        """ """

        # noise variances as tensor, shape (F, 1, N)
        Vb_t = self.np2tensor(self.Vb).to(self.device)[:,None,:]

        self.g_t.data = torch.clip(self.g_t, min = 0.001)

        def closure():
            # reset gradients
            self.optimizer.zero_grad()
            # compute speech variance
            Vs_t = self.compute_Vs(self.Z_t)


            # compute likelihood variance
            Vx_t = self.g_t*Vs_t + Vb_t
            # Vx_t = Vs_t + Vb_t
            # compute prior params
            z_0 = torch.zeros(self.z_dim,1,1).to(self.device)
            z_tm1 = torch.cat([z_0, self.Z_t[:,:,:-1]], -1) # (L,1,F)

            _, self.z_mean_p, self.z_logvar_p = self.vae.generation_z(z_tm1.permute(-1,1,0), self.visual)
            self.z_mean_p = self.z_mean_p.permute(-1,1,0)
            self.z_logvar_p = self.z_logvar_p.permute(-1,1,0)


            # compute loss, do backward and update latent variables
            loss = ( (self.rec_power)*torch.mean(self.X_abs_2_t/Vx_t + torch.log(Vx_t)) +
                   torch.mean(self.b*(self.g_t) - (self.a-1)*torch.log((self.g_t))) +
                    (1-self.rec_power)*torch.mean((self.Z_t-self.z_mean_p).pow(2)/self.z_logvar_p.exp())
                   )

            loss.backward()

            return loss

        # Some optimization algorithms such as Conjugate Gradient and LBFGS
        # need to reevaluate the function multiple times, so we have to pass
        # in a closure that allows them to recompute our model. The closure
        # should clear the gradients, compute the loss, and return it.

        pbar = np.arange(self.num_E_step)
        for epoch in pbar:
            self.optimizer.step(closure)

        # compute variances
        self.compute_Vs(self.Z_t)
        self.compute_Vs_scaled()
        self.compute_Vx()


        WFs, WFn = self.compute_WF(sample=True)
        self.S_hat = WFs*self.X
        last_s_hat = self.npc2tensor(self.S_hat.transpose()).unsqueeze(1).to(self.device)

        # update numpy array from the new tensor
        self.Z = self.tensor2np(self.Z_t.detach())

        self.g_t.data = torch.clip(self.g_t, min = 0.001)
        self.g = self.tensor2np(self.g_t.detach())

    def compute_WF(self, sample=False):

        # compute Wiener Filters
        self.Vx_transpose = np.transpose(self.Vx, (1,0,-1))
        WFs = (self.Vs_scaled/self.Vx_transpose).squeeze()
        WFn = (self.Vb[:,None,:]/self.Vx_transpose).squeeze()

        return WFs, WFn


#%% Langevin Dynamics GDPEEM.

class LDGDPEEM(DEM):

    def __init__(self, X, Vf, W, H, g, Z, vae, num_iter, device, lr=1e-2, visual = None, is_z_oracle = False, is_noise_oracle = False, fix_gain = True,
                 num_E_step=1, verbose = False, Z_oracle = None, rec_power = 0.99):

        super().__init__(X=X, Vf = Vf, W=W, H=H, g=g, vae=vae, num_iter=num_iter,
             device=device, fix_gain = True)

        self.verbose = verbose
        # mixture power spectrogram as tensor, shape (F, N)
        self.X_abs_2_t = self.np2tensor(self.X_abs_2)[:,None,:].to(self.device)

        self.lr = 5e-4
        # intial value for the latent variables
        self.g_t = self.np2tensor(self.g).to(self.device)[None,None,:]
        self.Z = Z # shape (L, 1, N)
        self.Z_t = self.np2tensor(self.Z).to(self.device)
        if Z_oracle is not None:
            self.Z_oracle = Z_oracle # shape (L, N)
            self.Z_oracle_t = self.np2tensor(self.Z_oracle).to(self.device)

        self.visual = visual

        self.z_dim = Z.shape[0]
        # optimizer for the E-step
        self.optimizer = optim.SGD([self.Z_t.requires_grad_()]+ [self.g_t.requires_grad_()], lr=1., momentum=0.)  # momentum is set to zero  
        
        # VERY IMPORTANT PARAMETERS: If 1, bad results in the FFNN VAE case
        self.num_E_step = num_E_step

        self.vae.eval() # vae in eval mode

        self.is_z_oracle = is_z_oracle
        self.is_noise_oracle = is_noise_oracle
        self.a = 1.1
        self.b = 1.1

        self.rec_power = 0.5 
        
    def _noise(self, params): 
        """We are adding param+noise to each param."""
        std = np.sqrt(2 * self.lr)
        loss = 0.
        for param in params:
            noise = torch.randn_like(param) * std
            loss += (noise * param).sum()
        return loss
    
    def compute_Vs(self, Z):
        """ Z: tensor of shape (N, L) """

        Vs_t= self.vae.generation_x(Z.permute(-1,1,0), self.visual).permute(-1,1,0) # (F, 1 ,N)

        self.Vs_t = Vs_t.clone().detach()
        self.Vs = self.tensor2np(Vs_t.detach())

        return Vs_t


    def E_step(self):
        """ """

        # noise variances as tensor, shape (F, 1, N)
        Vb_t = self.np2tensor(self.Vb).to(self.device)[:,None,:]

        self.g_t.data = torch.clip(self.g_t, min = 0.001)

        def LD():
            # reset gradients
            self.optimizer.zero_grad()
            # compute speech variance
            Vs_t = self.compute_Vs(self.Z_t)


            # compute likelihood variance
            Vx_t = self.g_t*Vs_t + Vb_t
            # compute prior params
            z_0 = torch.zeros(self.z_dim,1,1).to(self.device)
            z_tm1 = torch.cat([z_0, self.Z_t[:,:,:-1]], -1) # (L,1,F)

            _, self.z_mean_p, self.z_logvar_p = self.vae.generation_z(z_tm1.permute(-1,1,0), self.visual)
            self.z_mean_p = self.z_mean_p.permute(-1,1,0)
            self.z_logvar_p = self.z_logvar_p.permute(-1,1,0)


            # compute loss, do backward and update latent variables
            loss = ( (self.rec_power)*torch.mean(self.X_abs_2_t/Vx_t + torch.log(Vx_t)) +
                   torch.mean(self.b*(self.g_t) - (self.a-1)*torch.log((self.g_t))) +
                    (1-self.rec_power)*torch.mean((self.Z_t-self.z_mean_p).pow(2)/self.z_logvar_p.exp())
                   ) * self.lr

            loss += self._noise([self.Z_t] + [self.g_t])  # add noise*param before calling backward!
            loss.backward()
            self.optimizer.step()
            
            return loss

        pbar = np.arange(self.num_E_step)
        for epoch in pbar:
            loss_ = LD()

        self.g_t.data = torch.clip(self.g_t, min = 0.001)
        self.g = self.tensor2np(self.g_t.detach())
        
        # update numpy array from the new tensor
        self.Z = self.tensor2np(self.Z_t.detach())
        
        # compute variances
        self.compute_Vs(self.Z_t)
        self.compute_Vs_scaled()
        self.compute_Vx()

        WFs, WFn = self.compute_WF(sample=True)
        self.S_hat = WFs*self.X
        last_s_hat = self.npc2tensor(self.S_hat.transpose()).unsqueeze(1).to(self.device)

    def compute_WF(self, sample=False):

        # compute Wiener Filters
        self.Vx_transpose = np.transpose(self.Vx, (1,0,-1))
        WFs = (self.Vs_scaled/self.Vx_transpose).squeeze()
        WFn = (self.Vb[:,None,:]/self.Vx_transpose).squeeze()

        return WFs, WFn
