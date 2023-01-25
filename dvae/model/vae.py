#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt

The code in this file is based part on the source code of:
- Simon Legaive (simon.leglaive@centralesupelec.fr)
- in “A recurrent variational autoencoder for speech enhancement” ICASSP, 2020

Adapted and modified by Mostafa Sadeghi (mostafa.sadeghi@inria.fr) and Ali Golmakani (golmakani77@yahoo.com).

"""

from torch import nn
import torch
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F

class myFlatten(nn.Module):
    def forward(self, input):
        return input.reshape(-1,input.size(-1)).t()

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


def build_VAE(cfg, device='cpu', vae_mode = 'A-VAE', exp_mode = 'test', v_vae_path = None):

    ### Load parameters
    # General
    x_dim = cfg.getint('Network', 'x_dim')
    z_dim = cfg.getint('Network','z_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')

    # Inference and generation
    dense_x_z = [] if cfg.get('Network', 'dense_x_z') == '' else [int(i) for i in cfg.get('Network', 'dense_x_z').split(',')]

    # Beta-vae
    beta = cfg.getfloat('Training', 'beta')

    # Build model
    if vae_mode == 'A-VAE':
        model = A_VAE(x_dim=x_dim, z_dim=z_dim,
                    dense_x_z=dense_x_z, activation=activation,
                    dropout_p=dropout_p, beta=beta, device=device, lognormal = False).to(device)

    elif vae_mode == 'V-VAE':
        model = V_VAE(x_dim=x_dim, z_dim=z_dim,
                    dense_x_z=dense_x_z, activation=activation,
                    dropout_p=dropout_p, beta=beta, device=device).to(device)

    elif vae_mode == 'AV-VAE':

        model = AV_VAE(x_dim=x_dim, z_dim=z_dim,
            dense_x_z=dense_x_z, activation=activation,
            dropout_p=dropout_p, beta=beta, device=device).to(device)

        if exp_mode == 'train' and v_vae_path:

            print("Loading Pretrained V-VAE")

            model_v_vae = V_VAE(x_dim=x_dim, z_dim=z_dim,
                        dense_x_z=dense_x_z, activation=activation,
                        dropout_p=dropout_p, beta=beta, device=device).to(device)

            model_v_vae.load_state_dict(torch.load(v_vae_path, map_location=device))

            model.vfeats = model_v_vae.vfeats

    elif vae_mode == 'AV-CVAE':

        model = AV_CVAE(x_dim=x_dim, z_dim=z_dim,
                    dense_x_z=dense_x_z, activation=activation,
                    dropout_p=dropout_p, beta=beta, device=device).to(device)

        if exp_mode == 'train' and v_vae_path:

            print("Loading Pretrained V-VAE")

            model_av_vae = AV_VAE(x_dim=x_dim, z_dim=z_dim,
                        dense_x_z=dense_x_z, activation=activation,
                        dropout_p=dropout_p, beta=beta, device=device).to(device)

            model.vfeats = model_av_vae.vfeats

    return model


#%% Standard audio-only VAE (A-VAE) model

class A_VAE(nn.Module):

    '''
    VAE model class
    x: input data
    z: latent variables
    y: output data
    hidden_dim_enc: python list, the dimensions of hidden layers for encoder,
                        its reverse is the dimensions of hidden layers for decoder
    '''

    def __init__(self, x_dim=None, z_dim=16,
                 dense_x_z=[128], activation='tanh',
                 dropout_p = 0, beta=1, device='cpu', lognormal = False):

        super().__init__()
        ### General parameters for storn
        self.x_dim = x_dim
        self.y_dim = self.x_dim
        self.z_dim = z_dim
        self.dropout_p = dropout_p
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')
        ### Inference
        self.dense_x_z = dense_x_z
        ### Generation
        self.dense_z_x = list(reversed(dense_x_z))
        ### Beta-loss
        self.beta = beta
        self.lognormal = lognormal

        self.build()


    def build(self):


        ###################
        #### Inference ####
        ###################
        # 1. x_t to z_t

        dic_layers = OrderedDict()
        if len(self.dense_x_z) == 0:
            dim_x_z = self.dim_x
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_x_z = self.dense_x_z[-1]
            for n in range(len(self.dense_x_z)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.x_dim, self.dense_x_z[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_x_z[n-1], self.dense_x_z[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)

        self.mlp_x_z = nn.Sequential(dic_layers)
        self.inf_mean = nn.Linear(dim_x_z, self.z_dim)
        self.inf_logvar = nn.Linear(dim_x_z, self.z_dim)

        ######################
        #### Generation x ####
        ######################
        # 1. z_t to x_t

        dic_layers = OrderedDict()
        if len(self.dense_z_x) == 0:
            dim_z_x = self.dim_z
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_z_x = self.dense_z_x[-1]
            for n in range(len(self.dense_z_x)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.z_dim, self.dense_z_x[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_z_x[n-1], self.dense_z_x[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)

        self.mlp_z_x = nn.Sequential(dic_layers)
        self.gen_logvar = nn.Linear(dim_z_x, self.y_dim)


    def zprior(self, v):
        #v = v.permute(1,2,0)
        #v = v[None,None,:,:,:]

        mean = torch.zeros(v.shape[0], self.z_dim)
        logvar = torch.zeros(v.shape[0], self.z_dim)
        z = torch.zeros(v.shape[0], self.z_dim)

        return mean, logvar, z

    def reparameterization(self, mean, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return torch.addcmul(mean, eps, std)


    def inference(self, x, v = 'NaN'):

        x_z = self.mlp_x_z(x)
        z_mean = self.inf_mean(x_z)
        z_logvar = self.inf_logvar(x_z)
        z = self.reparameterization(z_mean, z_logvar)
        if self.lognormal:
            z = torch.exp(z)
        return z, z_mean, z_logvar


    def generation_x(self, z, v = 'NaN'):

        z_x = self.mlp_z_x(z)
        log_y = self.gen_logvar(z_x)
        y = torch.exp(log_y)
        return y


    def forward(self, x, v = 'NaN', compute_loss=False, exp_mode = 'train'):

        # train input: (batch_size, x_dim, seq_len)
        # test input:  (x_dim, seq_len)
        # need input:  (seq_len, batch_size, x_dim)


        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        #x = x.permute(-1, 0, 1)

        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # main part
        z, z_mean, z_logvar = self.inference(x)

        if exp_mode == 'train':
            y = self.generation_x(z)

        elif exp_mode == 'test':
            y = self.generation_x(z_mean)

        # calculate loss
        if compute_loss:
            loss_tot, loss_recon, loss_KLD = self.get_loss(x, y, z_mean, z_logvar, batch_size, seq_len, self.beta)
            self.loss = (loss_tot, loss_recon, loss_KLD)

        # output of NN:    (seq_len, batch_size, dim)
        # output of model: (batch_size, dim, seq_len) or (dim, seq_len)

        self.y = y.squeeze()

        return self.y


    def get_loss(self, x, y, z_mean, z_logvar, batch_size, seq_len, beta=1):

        loss_recon = torch.sum( x/y + torch.log(y) )
        loss_KLD = -0.5 * torch.sum(z_logvar -  z_logvar.exp() - z_mean.pow(2))

        loss_recon = loss_recon / (batch_size * seq_len)
        loss_KLD = loss_KLD / (batch_size * seq_len)
        loss_tot = loss_recon + beta * loss_KLD

        return loss_tot, loss_recon, loss_KLD


    def get_info(self):

        info = []
        info.append("----- Inference -----")
        for layer in self.mlp_x_z:
            info.append(str(layer))

        info.append("----- Bottleneck -----")
        info.append(str(self.inf_mean))
        info.append(str(self.inf_logvar))

        info.append("----- Decoder -----")
        for layer in self.mlp_z_x:
            info.append(str(layer))
        info.append(str(self.gen_logvar))

        return info

#%% A standard visual-only VAE (V-VAE) model that does not leverage audio information

class V_VAE(nn.Module):

    '''
    VAE model class
    x: input data
    z: latent variables
    y: output data
    hidden_dim_enc: python list, the dimensions of hidden layers for encoder,
                        its reverse is the dimensions of hidden layers for decoder
    '''

    def __init__(self, x_dim=None, z_dim=16,
                 dense_x_z=[128], activation='tanh',
                 dropout_p = 0, beta=1, device='cpu'):

        super().__init__()
        ### General parameters for storn
        self.x_dim = x_dim
        self.y_dim = self.x_dim
        self.z_dim = z_dim
        self.dropout_p = dropout_p
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')
        ### Inference
        self.dense_x_z = dense_x_z
        ### Generation
        self.dense_z_x = list(reversed(dense_x_z))
        ### Beta-loss
        self.beta = beta

        self.decoder_layers = nn.ModuleList()

        self.build()


    def build(self):

        ###################
        #### Inference ####
        ###################
        # 1. x_t to z_t

        #### Define encoder ####

        self.vfeats = nn.Sequential(
                        nn.Conv3d(1, 8, (9,9,5), stride=(2,2,1), padding=(0,0,2), dilation=1),
                        nn.ReLU(),
                        nn.Conv3d(8, 16, (5,5,5), stride=(2,2,1), padding=(0,0,2), dilation=1),
                        nn.ReLU(),
                        nn.Conv3d(16, 32, (3,3,5), stride=(2,2,1), padding=(0,0,2), dilation=1),
                        nn.ReLU(),
                        nn.Conv3d(32, 32, (3,3,5), stride=(2,2,1), padding=(0,0,2), dilation=1),
                        nn.ReLU(),
                        Flatten(),
                        nn.Linear(640, 64),
                        nn.ReLU()
                        )
        #### Define bottleneck layer ####

        self.latent_mean_layer = nn.Linear(64, self.z_dim)
        self.latent_logvar_layer = nn.Linear(64, self.z_dim)

        #### Define decoder ####

        hidden_dim_decoder = self.dense_z_x
        for n, dim in enumerate(hidden_dim_decoder):
            if n==0:
                self.decoder_layers.append(nn.Linear(self.z_dim, dim))
            else:
                self.decoder_layers.append(nn.Linear(hidden_dim_decoder[n-1],
                                                     dim))


        self.output_layer = nn.Linear(hidden_dim_decoder[-1], self.x_dim)

    def reparameterization(self, mean, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return torch.addcmul(mean, eps, std)


    def inference(self, x, v = 'NaN'):
        v = v.permute(0,4,2,3,1)
        # v = v[None,None,:,:,:]

        v = self.vfeats(v)

        mean = self.latent_mean_layer(v)
        logvar = self.latent_logvar_layer(v)
        z = self.reparameterization(mean, logvar)

        return z, mean, logvar


    def generation_x(self, z, v = 'NaN'):

        for layer in self.decoder_layers:
            z = self.activation(layer(z))

        return torch.exp(self.output_layer(z))


    def forward(self, x, v = 'NaN', compute_loss=False):
        # train input: (batch_size, x_dim, seq_len)
        # test input:  (x_dim, seq_len)
        # need input:  (seq_len, batch_size, x_dim)


        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # main part
        z, z_mean, z_logvar = self.inference(x, v)
        y = self.generation_x(z)

        # calculate loss
        if compute_loss:
            loss_tot, loss_recon, loss_KLD = self.get_loss(x, y, z_mean, z_logvar, batch_size, seq_len, self.beta)
            self.loss = (loss_tot, loss_recon, loss_KLD)

        # output of NN:    (seq_len, batch_size, dim)
        # output of model: (batch_size, dim, seq_len) or (dim, seq_len)

        self.y = y.squeeze()

        return self.y


    def get_loss(self, x, y, z_mean, z_logvar, batch_size, seq_len, beta=1):

        loss_recon = torch.sum( x/y + torch.log(y) )
        loss_KLD = -0.5 * torch.sum(z_logvar -  z_logvar.exp() - z_mean.pow(2))

        loss_recon = loss_recon / (batch_size * seq_len)
        loss_KLD = loss_KLD / (batch_size * seq_len)
        loss_tot = loss_recon + beta * loss_KLD

        return loss_tot, loss_recon, loss_KLD


    def get_info(self):

        info = []
        info.append("----- Inference -----")
        for layer in self.vfeats:
            info.append(str(layer))

        info.append("----- Bottleneck -----")
        info.append(str(self.latent_mean_layer))
        info.append(str(self.latent_logvar_layer))

        info.append("----- Decoder -----")
        for layer in self.decoder_layers:
            info.append(str(layer))
        info.append(str(self.output_layer))

        return info

#%% An audio-visual VAE (AV-VAE) model that leverages both audio and visual information

class AV_VAE(nn.Module):

    '''
    VAE model class
    x: input data
    z: latent variables
    y: output data
    hidden_dim_enc: python list, the dimensions of hidden layers for encoder,
                        its reverse is the dimensions of hidden layers for decoder
    '''

    def __init__(self, x_dim=None, z_dim=16,
                 dense_x_z=[128], activation='tanh',
                 dropout_p = 0, beta=1, device='cpu'):

        super().__init__()
        ### General parameters for storn
        self.x_dim = x_dim
        self.y_dim = self.x_dim
        self.z_dim = z_dim
        self.dropout_p = dropout_p
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')
        ### Inference
        self.dense_x_z = dense_x_z
        ### Generation
        self.dense_z_x = list(reversed(dense_x_z))
        ### Beta-loss
        self.beta = beta

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        self.build()


    def build(self):

        ###################
        #### Inference ####
        ###################
        # 1. x_t to z_t

        #### Define encoder ####

        self.vfeats = nn.Sequential(
                        nn.Conv3d(1, 8, (9,9,5), stride=(2,2,1), padding=(0,0,2), dilation=1),
                        nn.ReLU(),
                        nn.Conv3d(8, 16, (5,5,5), stride=(2,2,1), padding=(0,0,2), dilation=1),
                        nn.ReLU(),
                        nn.Conv3d(16, 32, (3,3,5), stride=(2,2,1), padding=(0,0,2), dilation=1),
                        nn.ReLU(),
                        nn.Conv3d(32, 32, (3,3,5), stride=(2,2,1), padding=(0,0,2), dilation=1),
                        nn.ReLU(),
                        Flatten(),
                        nn.Linear(640, 64),
                        nn.ReLU()
                        # nn.Linear(67*67, 64),
                        # nn.ReLU()


                        )

        dic_layers = OrderedDict()
        if len(self.dense_x_z) == 0:
            dim_x_z = self.dim_x
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_x_z = self.dense_x_z[-1]
            for n in range(len(self.dense_x_z)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.x_dim, self.dense_x_z[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_x_z[n-1], self.dense_x_z[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)

        self.mlp_x_z = nn.Sequential(dic_layers)

        #### Define prior layer ####

        self.zprior_mean_layer = nn.Linear(64, self.z_dim)
        self.zprior_logvar_layer = nn.Linear(64, self.z_dim)

        #### Define bottleneck layer ####

        self.latent_mean_layer = nn.Linear(self.dense_x_z[-1], self.z_dim)
        self.latent_logvar_layer = nn.Linear(self.dense_x_z[-1], self.z_dim)

        #### Define decoder ####

        hidden_dim_decoder = self.dense_z_x
        for n, dim in enumerate(hidden_dim_decoder):
            if n==0:
                self.decoder_layers.append(nn.Linear(self.z_dim, dim))
            else:
                self.decoder_layers.append(nn.Linear(hidden_dim_decoder[n-1],
                                                     dim))


        self.output_layer = nn.Linear(hidden_dim_decoder[-1], self.x_dim)

    def zprior(self, v):
        v = v.permute(0,4,2,3,1)
        # v = v[None,None,:,:,:]
        v = self.vfeats(v)
        mean = self.zprior_mean_layer(v)
        logvar = self.zprior_logvar_layer(v)
        z = self.reparameterization(mean, logvar)

        return mean, logvar, z

    def reparameterization(self, mean, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return torch.addcmul(mean, eps, std)


    def inference(self, x, v = 'NaN'):
        x = self.mlp_x_z(x)

        mean = self.latent_mean_layer(x)
        logvar = self.latent_logvar_layer(x)
        z = self.reparameterization(mean, logvar)

        return z, mean, logvar


    def generation_x(self, z, v = 'NaN'):

        for layer in self.decoder_layers:
            z = self.activation(layer(z))

        return torch.exp(self.output_layer(z))


    def forward(self, x, v = 'NaN', compute_loss=False):

        # train input: (batch_size, x_dim, seq_len)
        # test input:  (x_dim, seq_len)
        # need input:  (seq_len, batch_size, x_dim)


        if len(x.shape) == 2:
            x = x.unsqueeze(0)

        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # main part
        zp_mean, zp_logvar, z_p = self.zprior(v)
        z, z_mean, z_logvar = self.inference(x)
        y = self.generation_x(z)
        y_zp = self.generation_x(z_p)

        # calculate loss
        if compute_loss:
            loss_tot, loss_recon, loss_KLD = self.get_loss(x, y, z_mean, z_logvar, y_zp, zp_mean, zp_logvar, batch_size, seq_len, self.beta)
            self.loss = (loss_tot, loss_recon, loss_KLD)

        # output of NN:    (seq_len, batch_size, dim)
        # output of model: (batch_size, dim, seq_len) or (dim, seq_len)

        self.y = y.squeeze()

        return self.y


    def get_loss(self, x, y, z_mean, z_logvar, y_zp, zp_mean, zp_logvar, batch_size, seq_len, beta=1, alpha=1.):

        loss_recon = torch.sum( x/y + torch.log(y) )

        loss_KLD = -0.5 * torch.sum(z_logvar-zp_logvar - (z_logvar.exp()+(z_mean-zp_mean).pow(2))/(zp_logvar.exp()))
        loss_recon_zp = torch.sum( x/y_zp - torch.log(x/y_zp) - 1)

        loss_recon = loss_recon / (batch_size * seq_len)
        loss_KLD = loss_KLD / (batch_size * seq_len)
        loss_recon_zp = loss_recon_zp / (batch_size * seq_len)
        loss_tot = alpha*(loss_recon + loss_KLD)+(1.-alpha)*loss_recon_zp

        return loss_tot, loss_recon, loss_KLD


    def get_info(self):

        info = []
        info.append("----- z-prior -----")
        for layer in self.vfeats:
            info.append(str(layer))

        info.append("----- Inference -----")
        for layer in self.mlp_x_z:
            info.append(str(layer))

        info.append("----- Bottleneck -----")
        info.append(str(self.latent_mean_layer))
        info.append(str(self.latent_logvar_layer))

        info.append("----- Decoder -----")
        for layer in self.decoder_layers:
            info.append(str(layer))
        info.append(str(self.output_layer))

        return info

#%% An audio-visual conditional VAE (AV-CVAE) model

class AV_CVAE(nn.Module):

    '''
    VAE model class
    x: input data
    z: latent variables
    y: output data
    hidden_dim_enc: python list, the dimensions of hidden layers for encoder,
                        its reverse is the dimensions of hidden layers for decoder
    '''

    def __init__(self, x_dim=None, z_dim=16,
                 dense_x_z=[128], activation='tanh',
                 dropout_p = 0, beta=1, device='cpu'):

        super().__init__()
        ### General parameters for storn
        self.x_dim = x_dim
        self.y_dim = self.x_dim
        self.z_dim = z_dim
        self.dropout_p = dropout_p
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')
        ### Inference
        self.dense_x_z = dense_x_z
        ### Generation
        self.dense_z_x = list(reversed(dense_x_z))
        ### Beta-loss
        self.beta = beta

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        ### Visual Features
        self.dim_visual_features = 512
        self.build()


    def build(self):

        ###################
        #### Inference ####
        ###################
        # 1. x_t to z_t

        #### Define encoder ####

#         self.vfeats = nn.Sequential(
#                         nn.Conv3d(1, 8, (9,9,5), stride=(2,2,1), padding=(0,0,2), dilation=1),
#                         nn.ReLU(),
#                         nn.Conv3d(8, 16, (5,5,5), stride=(2,2,1), padding=(0,0,2), dilation=1),
#                         nn.ReLU(),
#                         nn.Conv3d(16, 32, (3,3,5), stride=(2,2,1), padding=(0,0,2), dilation=1),
#                         nn.ReLU(),
#                         nn.Conv3d(32, 32, (3,3,5), stride=(2,2,1), padding=(0,0,2), dilation=1),
#                         nn.ReLU(),
#                         Flatten(),
#                         nn.Linear(640, 64),
#                         nn.ReLU()
#                         # nn.Linear(67*67, 64),
#                         # nn.ReLU()


#                         )


        self.encoder_layerV = nn.Linear(self.dim_visual_features, self.dense_x_z[-1])
        self.layer_mix_encoder = nn.Linear(2*self.dense_x_z[-1], self.dense_x_z[-1])

        self.dim_v_p = self.dense_x_z[-1]
        self.prior_layerV = nn.Linear(self.dim_visual_features, self.dim_v_p)


        self.decoder_layerV = nn.Linear(self.dim_visual_features, self.dense_z_x[-1])
        self.layer_mix_decoder = nn.Linear(2*self.dense_z_x[-1], self.dense_z_x[-1])

        dic_layers = OrderedDict()
        if len(self.dense_x_z) == 0:
            dic_layers['Identity'] = nn.Identity()
        else:
            for n in range(len(self.dense_x_z)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.x_dim, self.dense_x_z[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_x_z[n-1], self.dense_x_z[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)

        self.mlp_x_z = nn.Sequential(dic_layers)


        #### Define prior layer ####

        self.zprior_mean_layer = nn.Linear(self.dim_v_p, self.z_dim)
        self.zprior_logvar_layer = nn.Linear(self.dim_v_p, self.z_dim)

        #### Define bottleneck layer ####

        self.inf_mean = nn.Linear(self.dense_x_z[-1], self.z_dim)
        self.inf_logvar = nn.Linear(self.dense_x_z[-1], self.z_dim)

        #### Define decoder ####

        dic_layers = OrderedDict()
        if len(self.dense_z_x) == 0:
            dic_layers['Identity'] = nn.Identity()
        else:
            for n in range(len(self.dense_z_x)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.z_dim, self.dense_z_x[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_z_x[n-1], self.dense_z_x[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)

        self.mlp_z_x = nn.Sequential(dic_layers)
        self.gen_logvar = nn.Linear(self.dense_z_x[-1], self.x_dim)

    def zprior(self, v):
        # v = v.permute(0,4,2,3,1)
        # v = v[None,None,:,:,:]

        v = self.prior_layerV(v)
        v = self.activation(v)

        mean = self.zprior_mean_layer(v)
        logvar = self.zprior_logvar_layer(v)
        z = self.reparameterization(mean, logvar)

        return mean, logvar, z

    def reparameterization(self, mean, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return torch.addcmul(mean, eps, std)


    def inference(self, x, v = 'NaN', amask = False):
        # v = v.permute(0,4,2,3,1)
        # v = v[None,None,:,:,:]
        # v = self.vfeats(v)


        x = self.mlp_x_z(x)
        if amask:
            # zz.data = torch.zeros_like(zz.data)+0.001
            n_ind = 30
            mask_ratio = 0.1
            list_idx = sorted([np.random.randint(0,x.shape[-1]) for _ in range(n_ind-1)] + [x.shape[-1]])
            modidx = np.random.choice(range(n_ind), int(mask_ratio*n_ind), replace = False)
            # list_idx = sorted([np.random.randint(0,seq_len) for _ in range(19)] + [seq_len])
            # modidx = np.random.randint(0,20)
            list_idx2 = [i for i in range(x.shape[-1]) if sum(np.asarray(list_idx)<=i) % n_ind in modidx]
            x.data[...,list_idx2] = torch.zeros_like(x.data[...,list_idx2]) + 0.001


        v = self.encoder_layerV(v)
        v = self.activation(v)

        xv = torch.cat((x,v), 1)
        xv = self.layer_mix_encoder(xv)
        xv = self.activation(xv)

        mean = self.inf_mean(xv)
        logvar = self.inf_logvar(xv)
        z = self.reparameterization(mean, logvar)

        return z, mean, logvar


    def generation_x(self, z, v = 'NaN'):

        z = self.mlp_z_x(z)

        # v = v.permute(0,4,2,3,1)
        # v = v[None,None,:,:,:]
        # v = self.vfeats(v)

        v = self.decoder_layerV(v)
        v = self.activation(v)

        zv = torch.cat((z,v), 1)
        zv = self.layer_mix_decoder(zv)
        zv = self.activation(zv)

        return torch.exp(self.gen_logvar(zv))


    def forward(self, x, v = 'NaN', compute_loss=False, amask = False):

        # train input: (batch_size, x_dim, seq_len)
        # test input:  (x_dim, seq_len)
        # need input:  (seq_len, batch_size, x_dim)

        # x shape (B, F)
        # v shape (B, F)

        seq_len = 1
        batch_size = x.shape[0]

        # main part
        zp_mean, zp_logvar, z_p = self.zprior(v)

        z, z_mean, z_logvar = self.inference(x, v, amask = amask)
        y = self.generation_x(z, v)
        y_zp = self.generation_x(z_p, v)

        # calculate loss
        if compute_loss:
            loss_tot, loss_recon, loss_KLD = self.get_loss(x, y, z_mean, z_logvar, y_zp, zp_mean, zp_logvar, batch_size, seq_len, self.beta)
            self.loss = (loss_tot, loss_recon, loss_KLD)

        # output of NN:    (seq_len, batch_size, dim)
        # output of model: (batch_size, dim, seq_len) or (dim, seq_len)

        self.y = y.squeeze()

        return self.y


    def get_loss(self, x, y, z_mean, z_logvar, y_zp, zp_mean, zp_logvar, batch_size, seq_len, beta=1, alpha=0.9):

        loss_recon = torch.sum( x/y + torch.log(y) )
        loss_KLD = -0.5 * torch.sum(z_logvar-zp_logvar - (z_logvar.exp()+(z_mean-zp_mean).pow(2))/(zp_logvar.exp()))
        loss_recon_zp = torch.sum( x/y_zp - torch.log(x/y_zp) - 1)

        loss_recon = loss_recon / (batch_size * seq_len)
        loss_KLD = loss_KLD / (batch_size * seq_len)
        loss_recon_zp = loss_recon_zp / (batch_size * seq_len)
        loss_tot = alpha*(loss_recon + loss_KLD)+(1.-alpha)*loss_recon_zp

        return loss_tot, loss_recon, loss_KLD


    def get_info(self):

        info = []

        info.append("----- Inference -----")
        for layer in self.mlp_x_z:
            info.append(str(layer))

        info.append("----- Bottleneck -----")
        info.append(str(self.inf_mean))
        info.append(str(self.inf_logvar))

        info.append("----- Decoder -----")
        for layer in self.mlp_z_x:
            info.append(str(layer))
        info.append(str(self.gen_logvar))

        return info

if __name__ == '__main__':
    x_dim = 513
    device = 'cpu'
    vae = AV_VAE(x_dim = x_dim).to(device)
    model_info = vae.get_info()
    for i in model_info:
        print(i)
