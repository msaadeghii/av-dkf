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

    def __init__(self, X, Vf, W, H, g, vae, niter=100, device='cpu', verbose = False, fix_gain = False):

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
        self.niter = niter # number of iterations
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
        cost = np.zeros(self.niter)
        if spec_param is not None:
            if os.path.exists("temp"):
                shutil.rmtree("temp")
            os.makedirs("temp")
        t0 = time.time()
        list_cost = []
        list_score = []
        if tqdm is not None:
            pbar = tqdm(np.arange(self.niter))
        else:
            pbar = np.arange(self.niter)
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
                    # plt.show()
                    img2 = librosa.display.specshow(D2, y_axis='log', x_axis='time',

                                                   sr=sample_rate, ax= ax2)
                    # plt.show()
                    img3 = librosa.display.specshow(D3, y_axis='log', x_axis='time',

                                                   sr=sample_rate, ax= ax3)
                    # plt.show()
                    img4 = librosa.display.specshow(D4, y_axis='log', x_axis='time',

                                                   sr=sample_rate, ax= ax4)

                    ax5.plot([x[0] for x in list_score], label = "SDR", color="r")
                    ax6.plot([x[1] for x in list_score], label = "PESQ", color="g")
                    ax7.plot([x[2] for x in list_score], label = "STOI", color="b")
                    ax8.plot(list_cost, label="Cost")

                    spec_figure.savefig('temp/frame' + '%03d'%(n) + '.png')
                    plt.close(spec_figure)
                    # mozz

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

class DEM:
    """
    Meant to be an abstract class that should not be instantiated but only
    inherited.
    """

    def __init__(self, X, Vf, W, H, g, vae, niter=100, device='cpu', verbose = False, fix_gain = False):

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
        self.niter = niter # number of iterations
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
        cost = np.zeros(self.niter)
        if spec_param is not None:
            if os.path.exists("temp"):
                shutil.rmtree("temp")
            os.makedirs("temp")


        t0 = time.time()
        list_cost = []
        list_score = []
        if tqdm is not None:
            pbar = tqdm(np.arange(self.niter))
        else:
            pbar = np.arange(self.niter)
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
                    # plt.show()
                    img2 = librosa.display.specshow(D2, y_axis='log', x_axis='time',

                                                   sr=sample_rate, ax= ax2)
                    # plt.show()
                    img3 = librosa.display.specshow(D3, y_axis='log', x_axis='time',

                                                   sr=sample_rate, ax= ax3)
                    # plt.show()
                    img4 = librosa.display.specshow(D4, y_axis='log', x_axis='time',

                                                   sr=sample_rate, ax= ax4)

                    ax5.plot([x[0] for x in list_score], label = "SDR", color="r")
                    ax6.plot([x[1] for x in list_score], label = "PESQ", color="g")
                    ax7.plot([x[2] for x in list_score], label = "STOI", color="b")
                    ax8.plot(list_cost, label="Cost")

                    spec_figure.savefig('temp/frame' + '%03d'%(n) + '.png')
                    plt.close(spec_figure)
                    # mozz

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



#%%

class VEM(EM):

    def __init__(self, X, Vf, W, H, g, vae, device, niter=100, nsamples_E_step=1,
                 nsamples_WF=1, lr=1e-2, nepochs_E_step=10, batch_size=0, verbose = False):

        super().__init__(X=X, Vf = Vf, W=W, H=H, g=g, vae=vae, niter=niter,
             device=device)

        self.verbose = verbose
        self.nsamples_E_step = nsamples_E_step
        self.nsamples_WF = nsamples_WF

        # mixture power spectrogram as tensor, shape (F, N)
        self.X_abs_2_t = self.np2tensor(self.X_abs_2).to(self.device)

        # Define the encoder parameters as the ones to optimize during the
        # E-step
        self.params_to_optimize = []
        if type(vae).__name__ == 'VAE':
            for name in vae.named_parameters():
                if 'encoder' in name[0]:
                    self.params_to_optimize.append(name[1])
                elif 'latent' in name[0]:
                    self.params_to_optimize.append(name[1])
        elif type(vae).__name__ == 'RVAE':
            for name in vae.named_parameters():
                if 'enc_' in name[0]:
                    self.params_to_optimize.append(name[1])
        else:
            raise NameError('Unknown VAE type')

        self.optimizer = optim.Adam(self.params_to_optimize, lr=lr) # optimizer

        # VERY IMPORTANT PARAMETERS: If 1, bad results in the FFNN VAE case
        self.nepochs_E_step = nepochs_E_step

        self.vae.train() # vae in train mode

    def sample_posterior(self, nsamples=5):
        N = self.X.shape[1]

        if hasattr(self.vae, 'latent_dim'):
            L = self.vae.latent_dim
        elif hasattr(self.vae, 'z_dim'):
            L = self.vae.z_dim

        Z_t = torch.zeros((N, nsamples, L))
        with torch.no_grad():
            for r in np.arange(nsamples):
                 _, _, Z_t[:,r,:] = self.vae.inference(torch.t(self.X_abs_2_t), self.Vf)
        return Z_t

    def compute_Vs(self, Z):
        """ Z: tensor of shape (N, R, L) """
        with torch.no_grad():
            Vs_t = self.vae.generation_x(Z, self.Vf) # (N, R, F)

        if len(Vs_t.shape) == 2:
            # shape is (N, F) but we need (N, R, F)
            Vs_t = Vs_t.unsqueeze(1) # add a dimension in axis 1

        self.Vs = np.moveaxis(self.tensor2np(Vs_t), 0, -1)

    def E_step(self):
        """
        - update the parameters of q(z | x ), i.e. the parameters of the
        encoder when the mixture spectrogram is fed as input
        - sample from q(z | x)
        - compute Vs, Vs_scaled and Vx """

        # vector of gain parameters as tensor, shape (, N)
        g_t = self.np2tensor(self.g).to(self.device)

        # noise variances as tensor, shape (F, N)
        Vb_t = self.np2tensor(self.Vb).to(self.device)

        def closure():
            # reset gradients
            self.optimizer.zero_grad()

            # forward pass in the vae with mixture spectrogram as input
            Vs_t, mean_t, logvar_t, _ = self.vae(torch.t(self.X_abs_2_t))
            Vs_t = torch.t(Vs_t) # tensor of shape (F, N)
            mean_t = torch.t(mean_t) # tensor of shape (L, N)
            logvar_t = torch.t(logvar_t) # tensor of shape (L, N)

            Vx_t = g_t*Vs_t + Vb_t # likelihood variances, tensor of shape
            # (F, N)

            # compute loss, do backward and update parameters
            loss = ( torch.sum(self.X_abs_2_t/Vx_t + torch.log(Vx_t)) -
                    0.5*torch.sum(logvar_t - mean_t.pow(2) - logvar_t.exp()) )
            loss.backward()
            return loss

        for epoch in np.arange(self.nepochs_E_step):

            self.optimizer.step(closure)

        # sample from posterior
        Z_t = self.sample_posterior(self.nsamples_E_step) # tensor of shape
        # (N, R, L)

        # compute variances
        self.compute_Vs(Z_t)
        self.compute_Vs_scaled()
        self.compute_Vx()

    def compute_WF(self, sample=False):

        if sample:
            # sample from posterior
            Z_t = self.sample_posterior(self.nsamples_WF) # tensor of shape
            # (N, R, L)

            # compute variances
            self.compute_Vs(Z_t)
            self.compute_Vs_scaled()
            self.compute_Vx()

        # compute Wiener Filters
        WFs = np.mean(self.Vs_scaled/self.Vx, axis=0)
        WFn = np.mean(self.Vb/self.Vx, axis=0)

        return WFs, WFn


#%%

class PEEM(EM):

    def __init__(self, X, Vf, W, H, g, Z, vae, niter, device, lr=1e-2,
                 nepochs_E_step=10, attention = False, verbose = False):

        super().__init__(X=X, Vf = Vf, W=W, H=H, g=g, vae=vae, niter=niter,
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
        self.nepochs_E_step = nepochs_E_step

        self.vae.eval() # vae in eval mode

        self.zmean = torch.tensor([0.0]).to(self.device)
        self.zlogvar = torch.tensor([0.0]).to(self.device)
        self.attention = attention

    def compute_Vs(self, Z):
        """ Z: tensor of shape (N, L) """
        if self.attention:
            context, _ = self.vae.attention(torch.t(Z), self.Vf)
            v_ = context
        else:
            v_ = self.Vf

        Vs_t= self.vae.generation_x(torch.t(Z), v_) # (F, N)
        Vs_t = Vs_t.t()
        self.Vs = self.tensor2np(Vs_t.detach())

        return Vs_t

    def loss_function(self, V_x, X_abs_2_tensor, z):

        neg_likelihood = torch.sum( torch.log(V_x) + X_abs_2_tensor / V_x )

        neg_prior = torch.sum((z-self.zmean).pow(2)/self.zlogvar.exp())

        return neg_likelihood + neg_prior

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
        for epoch in np.arange(self.nepochs_E_step):
            self.optimizer.step(closure)

        # update numpy array from the new tensor
        self.Z = self.tensor2np(self.Z_t.detach())

        # compute variances
        self.compute_Vs(self.Z_t)
        self.compute_Vs_scaled()
        self.compute_Vx()
        list_t = [self.Vx,self.Vs,self.Vs_scaled,self.X_abs_2,self.H,self.W]



    def compute_WF(self, sample=False):
        # sample parameter useless

        # compute Wiener Filters
        WFs = self.Vs_scaled/self.Vx
        WFn = self.Vb/self.Vx

        return WFs, WFn

#%%

class DPEEM(DEM):

    def __init__(self, X, Vf, W, H, g, Z, vae, niter, device, lr=1e-3,
                 nepochs_E_step=1, attention = False, verbose = False):

        super().__init__(X=X, Vf = Vf, W=W, H=H, g=g, vae=vae, niter=niter,
             device=device)

        self.verbose = verbose
        # mixture power spectrogram as tensor, shape (F, N)
        self.X_abs_2_t = self.np2tensor(self.X_abs_2)[:,None,:].to(self.device)

        # intial value for the latent variables
        self.Z = Z # shape (L, 1, N)
        self.Z_t = self.np2tensor(self.Z).to(self.device)
        self.z_dim = Z.shape[0]
        # optimizer for the E-step
        self.optimizer = optim.Adam([self.Z_t.requires_grad_()], lr=lr)

        # VERY IMPORTANT PARAMETERS: If 1, bad results in the FFNN VAE case
        self.nepochs_E_step = nepochs_E_step

        self.vae.eval() # vae in eval mode


        self.attention = attention

    def compute_Vs(self, Z):
        """ Z: tensor of shape (N, L) """
        if self.attention:
            context, _ = self.vae.attention(Z, self.Vf)
            v_ = context
        else:
            v_ = self.Vf
        
        Vs_t= self.vae.generation_x(Z.permute(-1,1,0), v_.unsqueeze(0).permute(-1, 0, 1)).permute(-1,1,0) # (F, 1 ,N)

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
        for epoch in np.arange(self.nepochs_E_step):
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
        self.Vx_transpose = np.transpose(self.Vx, (1,0,-1))
        WFs = (self.Vs_scaled/self.Vx_transpose).squeeze()
        WFn = (self.Vb[:,None,:]/self.Vx_transpose).squeeze()

        return WFs, WFn


class MCEM(EM):

    def __init__(self, X, Vf, W, H, g, Z, vae, niter, device, nsamples_E_step=10,
                 burnin_E_step=30, nsamples_WF=25, burnin_WF=75, var_RW=0.01, verbose = False):

        super().__init__(X=X, Vf = Vf, W=W, H=H, g=g, vae=vae, niter=niter,
             device=device)

        self.verbose = verbose

        if type(vae).__name__ == 'RVAE':
            raise NameError('MCEM algorithm only valid for FFNN VAE')

        self.Z = Z # Last draw of the latent variables, shape (L, N)
        self.nsamples_E_step = nsamples_E_step
        self.burnin_E_step = burnin_E_step
        self.nsamples_WF = nsamples_WF
        self.burnin_WF = burnin_WF
        self.var_RW = var_RW

        # mixture power spectrogram as tensor, shape (F, N)
        self.X_abs_2_t = self.np2tensor(self.X_abs_2).to(self.device)

        self.vae.eval() # vae in eval mode

        self.zmean, self.zlogvar, _ = self.vae.zprior(self.Vf)
        self.zmean, self.zlogvar = self.zmean.t().to(self.device), self.zlogvar.t().to(self.device)

    def sample_posterior(self, Z, nsamples=10, burnin=30):
        # Metropolis-Hastings

        F, N = self.X.shape

        if hasattr(self.vae, 'latent_dim'):
            L = self.vae.latent_dim
        elif hasattr(self.vae, 'z_dim'):
            L = self.vae.z_dim

        # random walk variance as tensor
        var_RM_t = torch.tensor(np.float32(self.var_RW))

        # latent variables sampled from the posterior
        Z_sampled_t = torch.zeros(N, nsamples, L)

        # intial latent variables as tensor, shape (L, N)
        Z_t = self.np2tensor(Z).to(self.device)
        # speech variances as tensor, shape (F, N)
        Vs_t = torch.t(self.vae.generation_x(torch.t(Z_t), self.Vf)).to(self.device)
        # vector of gain parameters as tensor, shape (, N)
        g_t = self.np2tensor(self.g).to(self.device)
        # noise variances as tensor, shape (F, N)
        Vb_t = self.np2tensor(self.Vb).to(self.device)
        # likelihood variances as tensor, shape (F, N)
        Vx_t = g_t*Vs_t + Vb_t

        cpt = 0
        averaged_acc_rate = 0
        for m in np.arange(nsamples+burnin):

            # random walk over latent variables
            Z_prime_t = Z_t + (torch.sqrt(var_RM_t)*torch.randn(L, N)).to(self.device)

            # compute associated speech variances
            Vs_prime_t = torch.t(self.vae.generation_x(torch.t(Z_prime_t), self.Vf)).to(self.device) # (F, N)
            Vs_prime_scaled_t = g_t*Vs_prime_t
            Vx_prime_t = Vs_prime_scaled_t + Vb_t

            # compute log of acceptance probability
            acc_prob = ( torch.sum(torch.log(Vx_t) - torch.log(Vx_prime_t) +
                                   (1/Vx_t - 1/Vx_prime_t)*self.X_abs_2_t, 0) +
                        .5*torch.sum( ((Z_t-self.zmean).pow(2) - (Z_prime_t-self.zmean).pow(2))/self.zlogvar.exp(), 0) )

            # accept/reject
            is_acc = torch.log(torch.rand(N)).to(self.device) < acc_prob
            averaged_acc_rate += ( torch.sum(is_acc).cpu().numpy()/
                                  np.prod(is_acc.shape)*100/(nsamples+burnin) )

            Z_t[:,is_acc] = Z_prime_t[:,is_acc]
            # update variances
            Vs_t = torch.t(self.vae.generation_x(torch.t(Z_t), self.Vf)).to(self.device)
            Vx_t = g_t*Vs_t + Vb_t

            if m > burnin - 1:
                Z_sampled_t[:,cpt,:] = torch.t(Z_t)
                cpt += 1

        #print('averaged acceptance rate: %f' % (averaged_acc_rate))

        return Z_sampled_t


    def compute_Vs(self, Z):
        """ Z: tensor of shape (N, R, L) """

        with torch.no_grad():
            Vs_t = self.vae.generation_x(Z, self.Vf[:,None,:]).to(self.device) # (N, R, F)

        if len(Vs_t.shape) == 2:
            # shape is (N, F) but we need (N, R, F)
            Vs_t = Vs_t.unsqueeze(1) # add a dimension in axis 1

        self.Vs = np.moveaxis(self.tensor2np(Vs_t), 0, -1)  # (R, F, N)


    def E_step(self):
        """
        """
        # sample from posterior
        Z_t = self.sample_posterior(self.Z, self.nsamples_E_step,
                                    self.burnin_E_step) # (N, R, L)

        # update last draw
        self.Z = self.tensor2np(torch.squeeze(Z_t[:,-1,:])).T


        # compute variances
        self.compute_Vs(Z_t)
        self.compute_Vs_scaled()
        self.compute_Vx()

    def compute_WF(self, sample=False):

        if sample:
            # sample from posterior
            Z_t = self.sample_posterior(self.Z, self.nsamples_WF,
                                        self.burnin_WF)

            # compute variances
            self.compute_Vs(Z_t)
            self.compute_Vs_scaled()
            self.compute_Vx()

        WFs = np.mean(self.Vs_scaled/self.Vx, axis=0)
        WFn = np.mean(self.Vb/self.Vx, axis=0)

        return WFs, WFn




class GPEEM(EM):

    def __init__(self, X, Vf, W, H, g, Z, vae, niter, device, lr=1e-2,
                 nepochs_E_step=10, attention = False, verbose = False):

        super().__init__(X=X, Vf = Vf, W=W, H=H, g=g, vae=vae, niter=niter,
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
        self.nepochs_E_step = nepochs_E_step

        self.vae.eval() # vae in eval mode

        self.zmean = torch.tensor([0.0]).to(self.device)
        self.zlogvar = torch.tensor([0.0]).to(self.device)
        self.attention = attention
        self.Z_old = self.Z_t

    def compute_Vs(self, Z):
        """ Z: tensor of shape (N, L) """
        if self.attention:
            context, _ = self.vae.attention(torch.t(Z), self.Vf)
            v_ = context
        else:
            v_ = self.Vf

        Vs_t= self.vae.generation_x(torch.t(Z), v_) # (F, N)
        Vs_t = Vs_t.t()
        self.Vs = self.tensor2np(Vs_t.detach())

        return Vs_t

    def loss_function(self, V_x, X_abs_2_tensor, z):

        neg_likelihood = torch.sum( torch.log(V_x) + X_abs_2_tensor / V_x )

        neg_prior = torch.sum((z-self.zmean).pow(2)/self.zlogvar.exp())

        return neg_likelihood + neg_prior

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
        for epoch in np.arange(self.nepochs_E_step):
            self.optimizer.step(closure)
            self.g_t.data = torch.clip(self.g_t.data, min=0.001)
        # update numpy array from the new tensor
        self.Z = self.tensor2np(self.Z_t.detach())

        # compute variances
        self.compute_Vs(self.Z_t)
        self.compute_Vs_scaled()
        self.compute_Vx()
        list_t = [self.Vx,self.Vs,self.Vs_scaled,self.X_abs_2,self.H,self.W]

        self.Z_old = self.Z_t.clone()
        self.g = self.tensor2np(self.g_t.detach())


    def compute_WF(self, sample=False):
        # sample parameter useless

        # compute Wiener Filters
        WFs = self.Vs_scaled/self.Vx
        WFn = self.Vb/self.Vx

        return WFs, WFn



class GDPEEM(DEM):

    def __init__(self, X, Vf, W, H, g, Z, vae, niter, device, lr=1e-3, visual = None, is_z_oracle = False, is_noise_oracle = False, fix_gain = False,
                 nepochs_E_step=1, attention = False, verbose = False, alpha = 0.95, Z_oracle = None, rec_power = 0.99):

        super().__init__(X=X, Vf = Vf, W=W, H=H, g=g, vae=vae, niter=niter,
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
        self.nepochs_E_step = nepochs_E_step

        self.vae.eval() # vae in eval mode

        self.is_z_oracle = is_z_oracle
        self.is_noise_oracle = is_noise_oracle
        self.attention = attention
        self.a = 1.1
        self.b = 1.1

        self.rec_power = rec_power

        self.iter_refresh = 0


    def compute_Vs(self, Z):
        """ Z: tensor of shape (N, L) """
        if self.attention:
            context, _ = self.vae.attention(Z, self.Vf)
            v_ = context
        else:
            v_ = self.Vf

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
            self.loss_value = np.mean(loss.detach().cpu().numpy())

            return loss

        # Some optimization algorithms such as Conjugate Gradient and LBFGS
        # need to reevaluate the function multiple times, so we have to pass
        # in a closure that allows them to recompute our model. The closure
        # should clear the gradients, compute the loss, and return it.

        # pbar = tqdm(np.arange(self.nepochs_E_step))
        pbar = np.arange(self.nepochs_E_step)
        for epoch in pbar:
            self.optimizer.step(closure)

        # compute variances
        self.compute_Vs(self.Z_t)
        self.compute_Vs_scaled()
        self.compute_Vx()


        WFs, WFn = self.compute_WF(sample=True)
        self.S_hat = WFs*self.X
        last_s_hat = self.npc2tensor(self.S_hat.transpose()).unsqueeze(1).to(self.device)


        self.iter_refresh += 1
        if self.iter_refresh % 10 == 0 and False:
            _, self.z_mean_p, self.z_logvar_p = self.vae.inference(self.X_abs_2_t.permute((-1,1,0)), self.v_tcn, vmask = False, amask = False)
            self.z_mean_p = self.z_mean_p.detach().permute(-1,1,0)
            self.z_logvar_p = self.z_logvar_p.detach().permute(-1,1,0)

            _, self.Z_t_toc, _ = self.vae.inference(last_s_hat, self.v_tcn, vmask = False, amask = False, inf_mask = True)

            self.Z_t.data = self.Z_t_toc.permute(-1, 1, 0).clone().data
            print("I did it")
        # update numpy array from the new tensor
        self.Z = self.tensor2np(self.Z_t.detach())

        self.g_t.data = torch.clip(self.g_t, min = 0.001)
        self.g = self.tensor2np(self.g_t.detach())


    def compute_WF(self, sample=False):
        # sample parameter useless

        # compute Wiener Filters
        self.Vx_transpose = np.transpose(self.Vx, (1,0,-1))
        WFs = (self.Vs_scaled/self.Vx_transpose).squeeze()
        WFn = (self.Vb[:,None,:]/self.Vx_transpose).squeeze()

        return WFs, WFn
