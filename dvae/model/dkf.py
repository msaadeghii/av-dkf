#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt

The code in this file is based on:
- “Deep Kalman Filter” arXiv, 2015, Rahul G.Krishnan et al.
- "Structured Inference Networks for Nonlinear State Space Models" AAAI, 2017, Rahul G.Krishnan et al.

DKF refers to the deep Markov model in the second paper, which has two possibilities:
- with only backwrad RNN in inference, it's a Deep Kalman Smoother (DKS),
- with bi-directional RNN in inference, it's a ST-LR

Adapted to AV-DKF by Mostafa Sadeghi (mostafa.sadeghi@inria.fr) and Ali Golmakani (golmakani77@yahoo.com).

"""


from torch import nn
import torch
from collections import OrderedDict

from lipreading.utils import load_json, save2npz
from lipreading.utils import showLR, calculateNorm2, AverageMeter
from lipreading.utils import load_model, CheckpointSaver
from lipreading.model import Lipreading

from modules.transformer import TransformerEncoder

import numpy as np

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def build_DKF(cfg, device='cuda', vae_mode = 'A-DKF'):
    ### Set Defaults
    defaults = {'gating_unit_layers' : 2, 'proposed_mean_layers' : 2, "inference" : "gated", "rnn" : "LSTM", "fusion": "dense", "fusion_depth": "early"}

    ### Load parameters
    # General
    x_dim = cfg.getint('Network', 'x_dim')
    z_dim = cfg.getint('Network','z_dim')
    activation = cfg.get('Network', 'activation')
    dropout_p = cfg.getfloat('Network', 'dropout_p')
    # Inference
    inference_type = cfg.get('Network', 'inference', fallback=defaults['inference'])
    rnn_cell = cfg.get('Network', 'rnn', fallback=defaults['rnn'])
    gating_unit_layers = cfg.getint('Network', 'gating_unit_layers', fallback=defaults['gating_unit_layers'])
    proposed_mean_layers = cfg.getint('Network', 'proposed_mean_layers', fallback=defaults['proposed_mean_layers'])
    dense_x_gx = [] if cfg.get('Network', 'dense_x_gx') == '' else [int(i) for i in cfg.get('Network', 'dense_x_gx').split(',')]
    dim_RNN_gx = cfg.getint('Network', 'dim_RNN_gx')
    num_RNN_gx = cfg.getint('Network', 'num_RNN_gx')
    bidir_gx = cfg.getboolean('Network', 'bidir_gx')
    dense_ztm1_g = [] if cfg.get('Network', 'dense_ztm1_g') == '' else [int(i) for i in cfg.get('Network', 'dense_ztm1_g').split(',')]
    dense_g_z = [] if cfg.get('Network', 'dense_g_z') == '' else [int(i) for i in cfg.get('Network', 'dense_g_z').split(',')]
    # Generation
    dense_z_x = [] if cfg.get('Network', 'dense_z_x') == '' else [int(i) for i in cfg.get('Network', 'dense_z_x').split(',')]

    fusion = cfg.get('Network', 'fusion', fallback = defaults['fusion'])
    fusion_depth = cfg.get('Network', 'fusion_depth', fallback = defaults['fusion_depth'])

    # Beta-vae
    beta = cfg.getfloat('Training', 'beta')

    # prior loss
    alpha = cfg.getfloat('Training', 'alpha')
    
    # Build model
    if vae_mode == 'A-DKF':

        model = DKF_A(x_dim=x_dim, z_dim=z_dim, activation=activation,
                    dense_x_gx=dense_x_gx, dim_RNN_gx=dim_RNN_gx,
                    num_RNN_gx=num_RNN_gx, bidir_gx=bidir_gx,
                    dense_ztm1_g=dense_ztm1_g, dense_g_z=dense_g_z,
                    dense_z_x=dense_z_x, gating_unit_layers=gating_unit_layers, proposed_mean_layers=proposed_mean_layers,
                    dropout_p=dropout_p, beta=beta, device=device, inference_type = inference_type, rnn_cell = rnn_cell).to(device)

    elif vae_mode == 'V-DKF':

        model = DKF_V(x_dim=x_dim, z_dim=z_dim, activation=activation,
                    dense_x_gx=dense_x_gx, dim_RNN_gx=dim_RNN_gx,
                    num_RNN_gx=num_RNN_gx, bidir_gx=bidir_gx,
                    dense_ztm1_g=dense_ztm1_g, dense_g_z=dense_g_z,
                    dense_z_x=dense_z_x,
                    dropout_p=dropout_p, beta=beta, device=device).to(device)
        
    elif vae_mode == 'AV-DKF':

        model = DKF_AV(x_dim=x_dim, z_dim=z_dim, activation=activation,
                    dense_x_gx=dense_x_gx, dim_RNN_gx=dim_RNN_gx,
                    num_RNN_gx=num_RNN_gx, bidir_gx=bidir_gx,
                    dense_ztm1_g=dense_ztm1_g, dense_g_z=dense_g_z,
                    dense_z_x=dense_z_x, fusion = fusion, fusion_depth = fusion_depth,
                    dropout_p=dropout_p, beta=beta, alpha=alpha, device=device).to(device)

    return model



class DKF_A(nn.Module):

    def __init__(self, x_dim, z_dim=16, activation='tanh',
                 dense_x_gx=[], dim_RNN_gx=128, num_RNN_gx=1, bidir_gx=False,
                 dense_ztm1_g=[], dense_g_z=[],
                 dense_z_x=[128,128], gating_unit_layers = 2, proposed_mean_layers = 2,
                 dropout_p = 0, beta=1, device='cuda', rnn_cell = "LSTM", inference_type = "normal"):

        super().__init__()
        ### General parameters
        self.x_dim = x_dim
        self.y_dim = x_dim
        self.z_dim = z_dim
        self.rnn_cell = rnn_cell
        self.dropout_p = dropout_p
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')
        self.device = device
        ### Inference
        self.dense_x_gx = dense_x_gx
        self.dim_RNN_gx = dim_RNN_gx
        self.num_RNN_gx = num_RNN_gx
        self.bidir_gx = bidir_gx
        self.dense_ztm1_g = dense_ztm1_g
        self.dense_g_z = dense_g_z
        self.inference_type = inference_type
        ### Generation x
        self.dense_z_x = dense_z_x
        ### Generation z
        self.gating_unit_layers = gating_unit_layers
        self.proposed_mean_layers = proposed_mean_layers

        ### Beta-loss
        self.beta = beta

        self.build()


    def build(self):

        ###################
        #### Inference ####
        ###################
        # 1. x_t to g_tˆx
        dic_layers = OrderedDict()
        if len(self.dense_x_gx) == 1 and self.dense_x_gx[0] == 0:
            dim_x_gx = self.x_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_x_gx = self.dense_x_gx[-1]
            for n in range(len(self.dense_x_gx)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.x_dim, self.dense_x_gx[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_x_gx[n-1], self.dense_x_gx[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)

        self.mlp_x_gx = nn.Sequential(dic_layers)

        if self.rnn_cell == "LSTM":
            self.rnn_gx = nn.LSTM(dim_x_gx, self.dim_RNN_gx, self.num_RNN_gx, bidirectional=self.bidir_gx)
        elif self.rnn_cell == "RNN":
            self.rnn_gx = nn.RNN(dim_x_gx, self.dim_RNN_gx, self.num_RNN_gx, bidirectional=self.bidir_gx)
        elif self.rnn_cell == "GRU":
            self.rnn_gx = nn.GRU(dim_x_gx, self.dim_RNN_gx, self.num_RNN_gx, bidirectional=self.bidir_gx)

        # 2. g_tˆx and z_tm1 to g_t
        dic_layers = OrderedDict()
        if len(self.dense_ztm1_g) == 1 and self.dense_ztm1_g[0] == 0:
            n=0
            dic_layers['linear_last'] = nn.Linear(self.z_dim, self.dim_RNN_gx)
            dic_layers['activation_last'] = self.activation
            dic_layers['dropout_last'+str(n)] = nn.Dropout(p=self.dropout_p)
        else:
            for n in range(len(self.dense_ztm1_g)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.z_dim, self.dense_ztm1_g[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_ztm1_g[n-1], self.dense_ztm1_g[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)

            dic_layers['linear_last'] = nn.Linear(self.dense_ztm1_g[-1], self.dim_RNN_gx)
            dic_layers['activation_last'] = self.activation
            dic_layers['dropout_last'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_ztm1_g = nn.Sequential(dic_layers)

        # 3. g_t to z_t
        dic_layers = OrderedDict()
        if len(self.dense_g_z) == 1 and self.dense_g_z[0] == 0:
            dim_g_z = self.dim_RNN_gx
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_g_z = self.dense_g_z[-1]
            for n in range(len(self.dense_g_z)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dim_RNN_gx, self.dense_g_z[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_g_z[n-1], self.dense_g_z[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_g_z = nn.Sequential(dic_layers)
        self.inf_mean = nn.Linear(dim_g_z, self.z_dim)
        self.inf_logvar = nn.Linear(dim_g_z, self.z_dim)

        ######################
        #### Generation z ####
        ######################
        # 1. Gating Unit
        dic_layers = OrderedDict()
        for i in range(self.gating_unit_layers - 1):
            dic_layers[f'linear{i+1}'] = nn.Linear(self.z_dim, self.z_dim)
            dic_layers['ReLU'] = nn.ReLU()
        dic_layers[f'linear{i+2}'] = nn.Linear(self.z_dim, self.z_dim)
        dic_layers['Sigmoid'] = nn.Sigmoid()
        self.mlp_gate = nn.Sequential(dic_layers)


        # 2. Proposed mean
        dic_layers = OrderedDict()
        for i in range(self.proposed_mean_layers - 1):
            dic_layers[f'linear{i+1}'] = nn.Linear(self.z_dim, self.z_dim)
            dic_layers['ReLU'] = nn.ReLU()
        dic_layers[f'linear{i+2}'] = nn.Linear(self.z_dim, self.z_dim)
        dic_layers['Sigmoid'] = nn.Sigmoid()
        self.mlp_z_prop = nn.Sequential(dic_layers)
        # 3. Prior
        self.prior_mean = nn.Linear(self.z_dim, self.z_dim)
        self.prior_logvar = nn.Sequential(nn.ReLU(),
                                          nn.Linear(self.z_dim, self.z_dim),
                                          nn.Softplus())

        ######################
        #### Generation x ####
        ######################
        dic_layers = OrderedDict()
        if len(self.dense_z_x) == 1 and self.dense_z_x[0] == 0:
            dim_z_x = self.z_dim
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


    def reparameterization(self, mean, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return torch.addcmul(mean, eps, std)


    def inference(self, x, v = None):

        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # Create variable holder and send to GPU if needed
        z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_logvar = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_t = torch.zeros((batch_size, self.z_dim)).to(self.device)

        # 1. x_t to g_t, g_t and z_tm1 to z_t
        x_g = self.mlp_x_gx(x)
        if self.bidir_gx:
            g, _ = self.rnn_gx(x_g)
            g = g.view(seq_len, batch_size, 2, self.dim_RNN_gx)
            g_forward = g[:,:,0,:]
            g_backward = g[:,:,1,:]
            for t in range(seq_len):
                g_t = (self.mlp_ztm1_g(z_t) + g_forward[t,:,:] + g_backward[t,:,:]) / 3
                g_z = self.mlp_g_z(g_t)
                z_mean[t,:,:] = self.inf_mean(g_z)
                z_logvar[t,:,:] = self.inf_logvar(g_z)
                z_t = self.reparameterization(z_mean[t,:,:], z_logvar[t,:,:])
                z[t,:,:] = z_t
        else:
            g, _ = self.rnn_gx(torch.flip(x_g, [0]))
            g = torch.flip(g, [0])
            for t in range(seq_len):
                g_t = (self.mlp_ztm1_g(z_t) + g[t,:,:]) / 2
                g_z = self.mlp_g_z(g_t)
                z_mean[t,:,:] = self.inf_mean(g_z)
                z_logvar[t,:,:] = self.inf_logvar(g_z)
                z_t = self.reparameterization(z_mean[t,:,:], z_logvar[t,:,:])
                z[t,:,:] = z_t

        return z, z_mean, z_logvar


    def generation_z(self, z_tm1, v=None):
        
        z_prop = self.mlp_z_prop(z_tm1)

        if self.inference_type == "gated":
            gate = self.mlp_gate(z_tm1)
            z_mean_p = (1 - gate) * self.prior_mean(z_tm1) + gate * z_prop
        elif self.inference_type == "normal":
            z_mean_p = self.prior_mean(z_prop)

        z_var_p = self.prior_logvar(z_prop)
        z_logvar_p = torch.log(z_var_p)

        return None, z_mean_p, z_logvar_p


    def generation_x(self, z, v = None):

        # 1. z_t to y_t
        log_y = self.mlp_z_x(z)
        log_y = self.gen_logvar(log_y)
        y = torch.exp(log_y)

        return y


    def forward(self, x, v = None, compute_loss=False):

        # train input: (batch_size, x_dim, seq_len)
        # test input:  (x_dim, seq_len)
        # need input:  (seq_len, batch_size, x_dim)
        
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        x = x.permute(-1, 0, 1)

        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # main part
        z, z_mean, z_logvar = self.inference(x, v)
        z_0 = torch.zeros(1, batch_size, self.z_dim).to(self.device)
        z_tm1 = torch.cat([z_0, z[:-1, :,:]], 0)
        _, z_mean_p, z_logvar_p = self.generation_z(z_tm1)
        y = self.generation_x(z)

        # calculate loss
        if compute_loss:
            loss_tot, loss_recon, loss_KLD = self.get_loss(x, y, z_mean, z_logvar,
                                                        z_mean_p, z_logvar_p,
                                                        seq_len, batch_size, self.beta)
            self.loss = (loss_tot, loss_recon, loss_KLD)

        # output of NN:    (seq_len, batch_size, dim)
        # output of model: (batch_size, dim, seq_len) or (dim, seq_len)
        self.y = y.permute(1,-1,0).squeeze()
        self.z = z.permute(1,-1,0).squeeze()
        self.z_mean = z_mean.permute(1,-1,0).squeeze()
        self.z_logvar = z_logvar.permute(1,-1,0).squeeze()
        self.z_mean_p = z_mean_p.permute(1,-1,0).squeeze()
        self.z_logvar_p = z_logvar_p.permute(1,-1,0).squeeze()

        return self.y


    def get_loss(self, x, y, z_mean, z_logvar, z_mean_p, z_logvar_p, seq_len, batch_size, beta=1):

        loss_recon = torch.sum( x/y + torch.log(y))
        loss_KLD = -0.5 * torch.sum(z_logvar - z_logvar_p
                - torch.div(z_logvar.exp() + (z_mean - z_mean_p).pow(2), z_logvar_p.exp()))

        loss_recon = loss_recon / (batch_size * seq_len)
        loss_KLD = loss_KLD / (batch_size * seq_len)
        loss_tot = loss_recon + beta * loss_KLD

        return loss_tot, loss_recon, loss_KLD


    def get_info(self):

        info = []
        info.append("----- Inference -----")
        info.append('>>>> x_t to g_t^x')
        for layer in self.mlp_x_gx:
            info.append(layer)
        info.append(self.rnn_gx)
        info.append('>>>> z_tm1 to g_x')
        info.append(self.mlp_ztm1_g)
        info.append('>>>> g_x to z_t')
        for layer in self.mlp_g_z:
            info.append(layer)

        info.append("----- Bottleneck -----")
        info.append(self.inf_mean)
        info.append(self.inf_logvar)

        info.append("----- Generation x -----")
        for layer in self.mlp_z_x:
            info.append(layer)
        info.append(self.gen_logvar)

        info.append("----- Generation z -----")
        info.append('>>>> Gating unit')
        for layer in self.mlp_gate:
            info.append(layer)
        info.append('>>>> Proposed mean')
        for layer in self.mlp_z_prop:
            info.append(layer)
        info.append('>>>> Prior mean and logvar')
        info.append(self.prior_mean)
        info.append(self.prior_logvar)

        return info

class DKF_V(nn.Module):

    def __init__(self, x_dim, z_dim=16, activation='tanh',
                 dense_x_gx=[256], dim_RNN_gx=128, num_RNN_gx=1, bidir_gx=False,
                 dense_ztm1_g=[32,64], dense_g_z=[64,32],
                 dense_z_x=[32,64,128,256], landmarks_dim = 67*67,
                 dropout_p = 0, beta=1, device='cpu'):

        super().__init__()
        ### General parameters
        self.landmarks_dim = landmarks_dim
        self.x_dim = x_dim
        self.y_dim = x_dim
        self.z_dim = z_dim
        self.dropout_p = dropout_p
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')

        self.activationV = nn.ReLU()

        self.device = device
        ### Inference
        self.dense_x_gx = dense_x_gx
        self.dim_RNN_gx = dim_RNN_gx
        self.num_RNN_gx = num_RNN_gx
        self.bidir_gx = bidir_gx
        self.dense_ztm1_g = dense_ztm1_g
        self.dense_g_z = dense_g_z
        ### Generation x
        self.dense_z_x = dense_z_x
        ### Beta-loss
        self.beta = beta

        self.build()


    def build(self):

        #### Define visual encoder ####

        # self.vfeats = nn.Sequential(
        #         nn.Linear(self.landmarks_dim, 512),
        #         nn.ReLU(),
        #         nn.Linear(512, self.x_dim),
        #         nn.ReLU()
        #         )

        # self.vfeats = nn.Sequential(
        #                 nn.Conv3d(1, 8, (9,9,5), stride=(2,2,1), padding=(0,0,2), dilation=1),
        #                 nn.ReLU(),
        #                 nn.Conv3d(8, 16, (5,5,5), stride=(2,2,1), padding=(0,0,2), dilation=1),
        #                 nn.ReLU(),
        #                 nn.Conv3d(16, 32, (3,3,5), stride=(2,2,1), padding=(0,0,2), dilation=1),
        #                 nn.ReLU(),
        #                 nn.Conv3d(32, 32, (3,3,5), stride=(2,2,1), padding=(0,0,2), dilation=1),
        #                 nn.ReLU(),
        #                 Flatten(),
        #                 nn.Linear(640, 64),
        #                 nn.ReLU()
        #                 )

        config_path = "/home/agolmakani/workarea/av-dvae/lrw_resnet18_mstcn.json"
        args_loaded = load_json(config_path)
        backbone_type = args_loaded['backbone_type']
        width_mult = args_loaded['width_mult']
        relu_type = args_loaded['relu_type']
        tcn_options = { 'num_layers': args_loaded['tcn_num_layers'],
                        'kernel_size': args_loaded['tcn_kernel_size'],
                        'dropout': args_loaded['tcn_dropout'],
                        'dwpw': args_loaded['tcn_dwpw'],
                        'width_mult': args_loaded['tcn_width_mult'],
                      }

        self.vfeats = Lipreading( modality='video',
                            num_classes=500,
                            tcn_options=tcn_options,
                            backbone_type=backbone_type,
                            relu_type=relu_type,
                            width_mult=width_mult,
                            extract_feats=True).cuda()
        self.vfeats = load_model("/home/agolmakani/workarea/av-dvae/lrw_resnet18_mstcn_adamw_s3.pth.tar", self.vfeats, allow_size_mismatch=False)
        self.vfeats.eval()



        ###################
        #### Inference ####
        ###################
        # 1. x_t to g_tˆx
        dic_layers = OrderedDict()
        if len(self.dense_x_gx) == 0:
            dim_x_gx = self.x_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_x_gx = self.dense_x_gx[-1]
            for n in range(len(self.dense_x_gx)):
                if n == 0:
                    # dic_layers['linear'+str(n)] = nn.Linear(self.x_dim, self.dense_x_gx[n])
                    dic_layers['linear'+str(n)] = nn.Linear(512, self.dense_x_gx[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_x_gx[n-1], self.dense_x_gx[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)

        self.mlp_x_gx = nn.Sequential(dic_layers)
        self.rnn_gx = nn.LSTM(dim_x_gx, self.dim_RNN_gx, self.num_RNN_gx, bidirectional=self.bidir_gx)

        # 2. g_tˆx and z_tm1 to g_t
        dic_layers = OrderedDict()
        if len(self.dense_ztm1_g) == 0:
            dic_layers['linear_last'] = nn.Linear(self.z_dim, self.dim_RNN_gx)
            dic_layers['activation_last'] = self.activation
            dic_layers['dropout_last'+str(n)] = nn.Dropout(p=self.dropout_p)
        else:
            for n in range(len(self.dense_ztm1_g)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.z_dim, self.dense_ztm1_g[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_ztm1_g[n-1], self.dense_ztm1_g[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)

            dic_layers['linear_last'] = nn.Linear(self.dense_ztm1_g[-1], self.dim_RNN_gx)
            dic_layers['activation_last'] = self.activation
            dic_layers['dropout_last'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_ztm1_g = nn.Sequential(dic_layers)

        # 3. g_t to z_t
        dic_layers = OrderedDict()
        if len(self.dense_g_z) == 0:
            dim_g_z = self.dim_RNN_gx
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_g_z = self.dense_g_z[-1]
            for n in range(len(self.dense_g_z)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dim_RNN_gx, self.dense_g_z[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_g_z[n-1], self.dense_g_z[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_g_z = nn.Sequential(dic_layers)
        self.inf_mean = nn.Linear(dim_g_z, self.z_dim)
        self.inf_logvar = nn.Linear(dim_g_z, self.z_dim)

        ######################
        #### Generation z ####
        ######################
        # 1. Gating Unit
        dic_layers = OrderedDict()
        dic_layers['linear1'] = nn.Linear(self.z_dim, self.z_dim)
        dic_layers['ReLU'] = nn.ReLU()
        dic_layers['linear2'] = nn.Linear(self.z_dim, self.z_dim)
        dic_layers['Sigmoid'] = nn.Sigmoid()
        self.mlp_gate = nn.Sequential(dic_layers)
        # 2. Proposed mean
        dic_layers = OrderedDict()
        dic_layers['linear1'] = nn.Linear(self.z_dim, self.z_dim)
        dic_layers['ReLU'] = nn.ReLU()
        dic_layers['linear2'] = nn.Linear(self.z_dim, self.z_dim)
        self.mlp_z_prop = nn.Sequential(dic_layers)
        # 3. Prior
        self.prior_mean = nn.Linear(self.z_dim, self.z_dim)
        self.prior_logvar = nn.Sequential(nn.ReLU(),
                                          nn.Linear(self.z_dim, self.z_dim),
                                          nn.Softplus())

        ######################
        #### Generation x ####
        ######################
        dic_layers = OrderedDict()
        if len(self.dense_z_x) == 0:
            dim_z_x = self.z_dim
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


    def reparameterization(self, mean, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return torch.addcmul(mean, eps, std)


    def inference(self, x, v = None):

        v = self.vfeats(v, lengths=None)
        v = v.permute(1,0,-1)

        seq_len = v.shape[0]
        batch_size = v.shape[1]

        # Create variable holder and send to GPU if needed
        z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_logvar = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_t = torch.zeros((batch_size, self.z_dim)).to(self.device)

        # 1. x_t to g_t, g_t and z_tm1 to z_t
        x_g = self.mlp_x_gx(v)
        if self.bidir_gx:
            g, _ = self.rnn_gx(x_g)
            g = g.view(seq_len, batch_size, 2, self.dim_RNN_gx)
            g_forward = g[:,:,0,:]
            g_backward = g[:,:,1,:]
            for t in range(seq_len):
                g_t = (self.mlp_ztm1_g(z_t) + g_forward[t,:,:] + g_backward[t,:,:]) / 3
                g_z = self.mlp_g_z(g_t)
                z_mean[t,:,:] = self.inf_mean(g_z)
                z_logvar[t,:,:] = self.inf_logvar(g_z)
                z_t = self.reparameterization(z_mean[t,:,:], z_logvar[t,:,:])
                z[t,:,:] = z_t
        else:
            g, _ = self.rnn_gx(torch.flip(x_g, [0]))
            g = torch.flip(g, [0])
            for t in range(seq_len):
                g_t = (self.mlp_ztm1_g(z_t) + g[t,:,:]) / 2
                g_z = self.mlp_g_z(g_t)
                z_mean[t,:,:] = self.inf_mean(g_z)
                z_logvar[t,:,:] = self.inf_logvar(g_z)
                z_t = self.reparameterization(z_mean[t,:,:], z_logvar[t,:,:])
                z[t,:,:] = z_t

        return z, z_mean, z_logvar


    def generation_z(self, z_tm1, v=None):

        gate = self.mlp_gate(z_tm1)
        z_prop = self.mlp_z_prop(z_tm1)
        z_mean_p = (1 - gate) * self.prior_mean(z_tm1) + gate * z_prop
        z_var_p = self.prior_logvar(z_prop)
        z_logvar_p = torch.log(z_var_p) # consistant with other models

        return None, z_mean_p, z_logvar_p


    def generation_x(self, z):

        # 1. z_t to y_t
        log_y = self.mlp_z_x(z)
        log_y = self.gen_logvar(log_y)
        y = torch.exp(log_y)

        return y


    def forward(self, x, v = None, compute_loss=False):

        # train input: (batch_size, x_dim, seq_len)
        # test input:  (x_dim, seq_len)
        # need input:  (seq_len, batch_size, x_dim)
        
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            v = v.unsqueeze(0)

        x = x.permute(-1, 0, 1)
        v = v.permute(0, 3, -1, 2, 1)

        seq_len = x.shape[0]
        batch_size = x.shape[1]

        # main part
        z, z_mean, z_logvar = self.inference(x, v)
        z_0 = torch.zeros(1, batch_size, self.z_dim).to(self.device)
        z_tm1 = torch.cat([z_0, z[:-1, :,:]], 0)
        z_mean_p, z_logvar_p = self.generation_z(z_tm1)
        y = self.generation_x(z)

        # calculate loss
        if compute_loss:
            loss_tot, loss_recon, loss_KLD = self.get_loss(x, y, z_mean, z_logvar,
                                                        z_mean_p, z_logvar_p,
                                                        seq_len, batch_size, self.beta)
            self.loss = (loss_tot, loss_recon, loss_KLD)

        # output of NN:    (seq_len, batch_size, dim)
        # output of model: (batch_size, dim, seq_len) or (dim, seq_len)
        self.y = y.permute(1,-1,0).squeeze()
        self.z = z.permute(1,-1,0).squeeze()
        self.z_mean = z_mean.permute(1,-1,0).squeeze()
        self.z_logvar = z_logvar.permute(1,-1,0).squeeze()
        self.z_mean_p = z_mean_p.permute(1,-1,0).squeeze()
        self.z_logvar_p = z_logvar_p.permute(1,-1,0).squeeze()

        return self.y


    def get_loss(self, x, y, z_mean, z_logvar, z_mean_p, z_logvar_p, seq_len, batch_size, beta=1):

        loss_recon = torch.sum( x/y + torch.log(y))
        loss_KLD = -0.5 * torch.sum(z_logvar - z_logvar_p
                - torch.div(z_logvar.exp() + (z_mean - z_mean_p).pow(2), z_logvar_p.exp()))

        loss_recon = loss_recon / (batch_size * seq_len)
        loss_KLD = loss_KLD / (batch_size * seq_len)
        loss_tot = loss_recon + beta * loss_KLD

        return loss_tot, loss_recon, loss_KLD


    def get_info(self):

        info = []
        info.append("----- Inference -----")
        info.append('>>>> x_t to g_t^x')
        for layer in self.mlp_x_gx:
            info.append(layer)
        info.append(self.rnn_gx)
        info.append('>>>> z_tm1 to g_x')
        info.append(self.mlp_ztm1_g)
        info.append('>>>> g_x to z_t')
        for layer in self.mlp_g_z:
            info.append(layer)

        info.append("----- Bottleneck -----")
        info.append(self.inf_mean)
        info.append(self.inf_logvar)

        info.append("----- Generation x -----")
        for layer in self.mlp_z_x:
            info.append(layer)
        info.append(self.gen_logvar)

        info.append("----- Generation z -----")
        info.append('>>>> Gating unit')
        for layer in self.mlp_gate:
            info.append(layer)
        info.append('>>>> Proposed mean')
        for layer in self.mlp_z_prop:
            info.append(layer)
        info.append('>>>> Prior mean and logvar')
        info.append(self.prior_mean)
        info.append(self.prior_logvar)

        return info



class DKF_AV(nn.Module):

    def __init__(self, x_dim, z_dim=16, activation='tanh',
                 dense_x_gx=[256], dim_RNN_gx=128, num_RNN_gx=1, bidir_gx=False,
                 dense_ztm1_g=[32,64], dense_g_z=[64,32],
                 dense_z_x=[32,64,128,256], landmarks_dim = 67*67, fusion="dense", fusion_depth = "early",
                 dropout_p = 0, beta=1, alpha = 0.99, device='cuda'):

        super().__init__()
        ### General parameters
        self.landmarks_dim = landmarks_dim
        self.x_dim = x_dim
        self.y_dim = x_dim
        self.z_dim = z_dim
        self.dropout_p = dropout_p
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise SystemExit('Wrong activation type!')

        self.device = device
        ### Inference
        self.dense_x_gx = dense_x_gx
        self.dim_RNN_gx = dim_RNN_gx
        self.num_RNN_gx = num_RNN_gx
        self.bidir_gx = bidir_gx
        self.dense_ztm1_g = dense_ztm1_g
        self.dense_g_z = dense_g_z
        ### Generation x
        self.dense_z_x = dense_z_x
        ### Beta-loss
        self.beta = beta

        ### Prior-loss
        self.alpha = alpha

        ### Visual Features
        self.fusion_depth = fusion_depth
        self.fusion = fusion
        self.dim_visual_features = 512
        
        self.build()

    def build(self):
        
        self.encoder_layerV = nn.Linear(self.dim_visual_features, self.dense_x_gx[-1])
        self.layer_mix_encoder = nn.Linear(2*self.dense_x_gx[-1], self.dense_x_gx[-1])

        self.prior_layerV = nn.Linear(self.dim_visual_features, self.z_dim)
        self.layer_mix_prior = nn.Linear(2*self.z_dim, self.z_dim)

        self.decoder_layerV = nn.Linear(self.dim_visual_features, self.dense_z_x[-1])
        self.layer_mix_decoder = nn.Linear(2*self.dense_z_x[-1], self.dense_z_x[-1])
        if self.fusion == "transformer":
            self.transformer_encoder = TransformerEncoder(32, 4, 1, pos_embed_dim = 16,
                                                                  query_dim = self.dense_x_gx[-1],
                                                                  key_dim = self.dense_x_gx[-1],
                                                                  value_dim = self.dense_x_gx[-1])
        ###################
        #### Inference ####
        ###################
        # 1. x_t to g_tˆx
        dic_layers = OrderedDict()
        if len(self.dense_x_gx) == 1 and self.dense_x_gx[0] == 0:
            dim_x_gx = self.x_dim
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_x_gx = self.dense_x_gx[-1]
            for n in range(len(self.dense_x_gx)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.x_dim, self.dense_x_gx[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_x_gx[n-1], self.dense_x_gx[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)

        self.mlp_x_gx = nn.Sequential(dic_layers)

        self.rnn_gx = nn.LSTM(dim_x_gx, self.dim_RNN_gx, self.num_RNN_gx, bidirectional=self.bidir_gx)

        # 2. g_tˆx and z_tm1 to g_t
        dic_layers = OrderedDict()
        if len(self.dense_ztm1_g) == 1 and self.dense_ztm1_g[0] == 0:
            dic_layers['linear_last'] = nn.Linear(self.z_dim, self.dim_RNN_gx)
            dic_layers['activation_last'] = self.activation
            dic_layers['dropout_last'+str(n)] = nn.Dropout(p=self.dropout_p)
        else:
            for n in range(len(self.dense_ztm1_g)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.z_dim, self.dense_ztm1_g[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_ztm1_g[n-1], self.dense_ztm1_g[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)

            dic_layers['linear_last'] = nn.Linear(self.dense_ztm1_g[-1], self.dim_RNN_gx)
            dic_layers['activation_last'] = self.activation
            dic_layers['dropout_last'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_ztm1_g = nn.Sequential(dic_layers)

        # 3. g_t to z_t
        dic_layers = OrderedDict()
        if len(self.dense_g_z) == 1 and self.dense_g_z[0] == 0:
            dim_g_z = self.dim_RNN_gx
            dic_layers['Identity'] = nn.Identity()
        else:
            dim_g_z = self.dense_g_z[-1]
            for n in range(len(self.dense_g_z)):
                if n == 0:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dim_RNN_gx, self.dense_g_z[n])
                else:
                    dic_layers['linear'+str(n)] = nn.Linear(self.dense_g_z[n-1], self.dense_g_z[n])
                dic_layers['activation'+str(n)] = self.activation
                dic_layers['dropout'+str(n)] = nn.Dropout(p=self.dropout_p)
        self.mlp_g_z = nn.Sequential(dic_layers)
        self.inf_mean = nn.Linear(dim_g_z, self.z_dim)
        self.inf_logvar = nn.Linear(dim_g_z, self.z_dim)

        ######################
        #### Generation z ####
        ######################
        # 1. Gating Unit
        dic_layers = OrderedDict()
        dic_layers['linear1'] = nn.Linear(self.z_dim, self.z_dim)
        dic_layers['ReLU'] = nn.ReLU()
        dic_layers['linear2'] = nn.Linear(self.z_dim, self.z_dim)
        dic_layers['Sigmoid'] = nn.Sigmoid()
        self.mlp_gate = nn.Sequential(dic_layers)
        # 2. Proposed mean
        dic_layers = OrderedDict()
        dic_layers['linear1'] = nn.Linear(self.z_dim, self.z_dim)
        dic_layers['ReLU'] = nn.ReLU()
        dic_layers['linear2'] = nn.Linear(self.z_dim, self.z_dim)
        self.mlp_z_prop = nn.Sequential(dic_layers)
        # 3. Prior
        self.prior_mean = nn.Linear(self.z_dim, self.z_dim)
        self.prior_logvar = nn.Sequential(nn.ReLU(),
                                          nn.Linear(self.z_dim, self.z_dim),
                                          nn.Softplus())

        ######################
        #### Generation x ####
        ######################
        dic_layers = OrderedDict()
        if len(self.dense_z_x) == 1 and self.dense_z_x[0] == 0:
            dim_z_x = self.z_dim
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


    def reparameterization(self, mean, logvar):

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return torch.addcmul(mean, eps, std)


    def inference(self, x, v = None, amask = False):

        x_g = self.mlp_x_gx(x)

        seq_len = v.shape[0]
        batch_size = v.shape[1]
        v_g = self.encoder_layerV(v)
        # self.v_g = self.activation(v_g)

        if self.fusion in ["transformer", "self_transformer"]:
            xv_g, self.att_weights = self.fill_z(x_g, v_g, amask=amask, inf_mask = False, mask_diag_margin = False, transformer = self.transformer_encoder,
                                                 activation = True, residual = False, fusion = self.fusion)
        elif self.fusion == "dense":
            if amask:
                n_ind = 30
                mask_ratio = 0.1
                list_idx = sorted([np.random.randint(0,seq_len) for _ in range(n_ind-1)] + [seq_len])
                modidx = np.random.choice(range(n_ind), int(mask_ratio*n_ind), replace = False)
                # list_idx = sorted([np.random.randint(0,seq_len) for _ in range(19)] + [seq_len])
                # modidx = np.random.randint(0,20)
                list_idx2= [i for i in range(seq_len) if sum(np.asarray(list_idx)<=i) % n_ind in modidx]
                x_g.data[list_idx2,...] = torch.zeros_like(x_g.data[list_idx2,...]) + 0.001

                # x_g.data = torch.zeros_like(x_g)

            xv_g = self.activation(self.layer_mix_encoder(torch.cat([x_g, v_g], -1)) + x_g)


        xv_g = torch.cat((x_g, v_g), axis=-1)
        xv_g = self.layer_mix_encoder(xv_g)
        xv_g = self.activation(xv_g)
        # xv_g = (1/2)*(x_g + v_g)

        # Create variable holder and send to GPU if needed
        z_mean = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_logvar = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z = torch.zeros((seq_len, batch_size, self.z_dim)).to(self.device)
        z_t = torch.zeros((batch_size, self.z_dim)).to(self.device)

        # 1. x_t to g_t, g_t and z_tm1 to z_t

        if self.bidir_gx:
            g, _ = self.rnn_gx(xv_g)
            g = g.view(seq_len, batch_size, 2, self.dim_RNN_gx)
            g_forward = g[:,:,0,:]
            g_backward = g[:,:,1,:]
            for t in range(seq_len):
                g_t = (self.mlp_ztm1_g(z_t) + g_forward[t,:,:] + g_backward[t,:,:]) / 3
                g_z = self.mlp_g_z(g_t)
                z_mean[t,:,:] = self.inf_mean(g_z)
                z_logvar[t,:,:] = self.inf_logvar(g_z)
                z_t = self.reparameterization(z_mean[t,:,:], z_logvar[t,:,:])
                z[t,:,:] = z_t
        else:
            g, _ = self.rnn_gx(torch.flip(xv_g, [0]))
            g = torch.flip(g, [0])
            for t in range(seq_len):
                g_t = (self.mlp_ztm1_g(z_t) + g[t,:,:]) / 2
                g_z = self.mlp_g_z(g_t)
                z_mean[t,:,:] = self.inf_mean(g_z)
                z_logvar[t,:,:] = self.inf_logvar(g_z)
                z_t = self.reparameterization(z_mean[t,:,:], z_logvar[t,:,:])
                z[t,:,:] = z_t

        return z, z_mean, z_logvar

    def fill_z(self, zz, v, amask=False, inf_mask = False, mask_diag_margin = 0, transformer = None, last_layer = None, activation = False, residual = False, fusion = "self_transformer", mask_ratio = 0.1):

        seq_len = v.shape[0]
        batch_size = v.shape[1]
        n_ind = 20

        list_idx2 = None
        if amask:
            
            list_idx = sorted([np.random.randint(0,seq_len) for _ in range(n_ind-1)] + [seq_len])
            modidx = np.random.choice(range(n_ind), int(mask_ratio*n_ind), replace = False)
            list_idx2= [i for i in range(seq_len) if sum(np.asarray(list_idx)<=i) % n_ind in modidx]
            zz.data[list_idx2,...] = torch.zeros_like(zz.data[list_idx2,...])+0.001
            if len(list_idx2) > 1:
                list_idx2 = list_idx2[1:]


        if fusion == "transformer":
            xv_g, att_weights = transformer(v, x_in_k = zz, x_in_v = zz, mask = list_idx2, mask_diag_margin = mask_diag_margin, inf_mask = inf_mask)
            if last_layer:
                xv_g = last_layer(xv_g)
            if residual:
                xv_g = xv_g + zz
            if activation:
                xv_g = self.activation(xv_g)

            # xv_g = self.activation(xv_g) + zz
            # xv_g = torch.cat((zz, v_g), axis=-1)
            # xv_g, att_weights = self.transformer_encoder(xv_g, mask = list_idx2)
        if fusion == "self_transformer":
            # xv_g, att_weights = self.transformer_encoder(v_g, x_in_k = zz, x_in_v = zz, mask = list_idx2)
            xv_g = torch.cat((zz, v), axis=-1)
            xv_g, att_weights = transformer(xv_g, mask = list_idx2, mask_diag_margin = mask_diag_margin, inf_mask = inf_mask)
            if last_layer:
                xv_g = last_layer(xv_g)
            if residual:
                xv_g = xv_g + zz
            if activation:
                xv_g = self.activation(xv_g)


        return xv_g, att_weights


    def generation_z(self, z_tm1, v):
        
        v_p = self.prior_layerV(v)
        v_p = self.activation(v_p)

        zv = torch.cat((z_tm1, v_p), axis=-1)
        zv = self.layer_mix_prior(zv)
        zv = self.activation(zv)

        gate = self.mlp_gate(zv)
        z_prop = self.mlp_z_prop(zv)
        z_mean_p = (1 - gate) * self.prior_mean(zv) + gate * z_prop
        z_var_p = self.prior_logvar(z_prop)
        z_logvar_p = torch.log(z_var_p) # consistant with other models
        z_p = self.reparameterization(z_mean_p, z_logvar_p)
        return z_p, z_mean_p, z_logvar_p


    def generation_x(self, z, v):
        
        v_p = self.decoder_layerV(v)
        v_p = self.activation(v_p)

        z_log_y = self.mlp_z_x(z)
        zv = torch.cat((z_log_y, v_p), axis=-1)
        zv = self.layer_mix_decoder(zv)
        zv = self.activation(zv)
        # zv = (1/2)*(z_log_y + v_p)

        log_y = self.gen_logvar(zv)
        y = torch.exp(log_y)

        return y


    def forward(self, x, v = None, compute_loss=False, amask = False):

        # train input: (batch_size, x_dim, seq_len)
        # test input:  (x_dim, seq_len)
        # need input:  (seq_len, batch_size, x_dim)
        
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            v = v.unsqueeze(0)

        x = x.permute(-1, 0, 1).float()
        v = v.permute(-1, 0, 1).float() 
        
        seq_len = x.shape[0]
        batch_size = x.shape[1]
        
        # main part
        z, z_mean, z_logvar = self.inference(x, v, amask = amask)
        z_0 = torch.zeros(1, batch_size, self.z_dim).to(self.device)
        z_tm1 = torch.cat([z_0, z[:-1, :,:]], 0)
        z_p, z_mean_p, z_logvar_p = self.generation_z(z_tm1, v)

        y_zp = self.generation_x(z_p, v)
        y = self.generation_x(z, v)

        # calculate loss
        if compute_loss:
            loss_tot, loss_recon, loss_KLD = self.get_loss(x, y, z_mean, z_logvar, y_zp,
                                                        z_mean_p, z_logvar_p,
                                                        seq_len, batch_size, self.beta, self.alpha)
            self.loss = (loss_tot, loss_recon, loss_KLD)

        # output of NN:    (seq_len, batch_size, dim)
        # output of model: (batch_size, dim, seq_len) or (dim, seq_len)
        self.y = y.permute(1,-1,0).squeeze()
        self.z = z.permute(1,-1,0).squeeze()
        self.z_mean = z_mean.permute(1,-1,0).squeeze()
        self.z_logvar = z_logvar.permute(1,-1,0).squeeze()
        self.z_mean_p = z_mean_p.permute(1,-1,0).squeeze()
        self.z_logvar_p = z_logvar_p.permute(1,-1,0).squeeze()

        return self.y


    def get_loss(self, x, y, z_mean, z_logvar, y_zp, z_mean_p, z_logvar_p, seq_len, batch_size, beta=1, alpha=0.99):

        loss_recon = torch.sum( x/y + torch.log(y))
        loss_KLD = -0.5 * torch.sum(z_logvar - z_logvar_p
                - torch.div(z_logvar.exp() + (z_mean - z_mean_p).pow(2), z_logvar_p.exp()))
        loss_recon_zp = torch.sum( x/y_zp + torch.log(y_zp))

        loss_recon = loss_recon / (batch_size * seq_len)
        loss_recon_zp = loss_recon_zp / (batch_size * seq_len)
        loss_KLD = loss_KLD / (batch_size * seq_len)
        loss_tot = loss_recon + beta * loss_KLD
        loss_tot = alpha*loss_tot + (1.-alpha)*loss_recon_zp


        return loss_tot, loss_recon, loss_KLD


    def get_info(self):

        info = []
        info.append("----- Inference -----")
        info.append('>>>> x_t to g_t^x')
        for layer in self.mlp_x_gx:
            info.append(layer)
        info.append(self.rnn_gx)
        info.append('>>>> z_tm1 to g_x')
        info.append(self.mlp_ztm1_g)
        info.append('>>>> g_x to z_t')
        for layer in self.mlp_g_z:
            info.append(layer)

        info.append("----- Bottleneck -----")
        info.append(self.inf_mean)
        info.append(self.inf_logvar)

        info.append("----- Generation x -----")
        for layer in self.mlp_z_x:
            info.append(layer)
        info.append(self.gen_logvar)

        info.append("----- Generation z -----")
        info.append('>>>> Gating unit')
        for layer in self.mlp_gate:
            info.append(layer)
        info.append('>>>> Proposed mean')
        for layer in self.mlp_z_prop:
            info.append(layer)
        info.append('>>>> Prior mean and logvar')
        info.append(self.prior_mean)
        info.append(self.prior_logvar)

        return info


if __name__ == '__main__':
    x_dim = 257
    z_dim = 16
    device = 'cpu'
    dkf = DKF_A(x_dim=x_dim, z_dim=z_dim).to(device)

    x = torch.ones((x_dim,50))
    v = torch.ones((4489,50))
    y = dkf(x, v)

    print(y.shape)
