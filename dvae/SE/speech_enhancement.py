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

my_seed = 0
import numpy as np
np.random.seed(my_seed)
import torch
torch.manual_seed(my_seed)
import soundfile as sf
import librosa
import librosa.display
from asteroid.metrics import get_metrics

import sys
from tqdm import tqdm
sys.path.append('dvae/model')
sys.path.append('dvae/utils')
sys.path.append('dvae/SE')
import os
from SE_algorithms import VEM, PEEM, DPEEM, MCEM, GDPEEM, GPEEM
from DSE_algorithms import DVEM
from vae import build_VAE
from dkf import build_DKF
from read_config import myconf
from scipy import signal
import torchvision.transforms.functional as Fresize
import matplotlib.pyplot as plt
import cv2

from lipreading.utils import load_json, save2npz
from lipreading.utils import showLR, calculateNorm2, AverageMeter
from lipreading.utils import load_model, CheckpointSaver
from lipreading.model import Lipreading


def resample(video, target_num):
    n, N = video.shape # (4489, 129)
    ratio = N / target_num
    idx_lst = np.arange(target_num).astype(float)
    idx_lst *= ratio
    res = np.zeros((n, target_num))
    for i in range(target_num):
        res[:,i] = video[:,int(idx_lst[i])]
    return res

class SpeechEnhancement:
    def __init__(self, saved_model, output_dir, nmf_rank = 8,
                 device = 'cpu', niter = 200, save_flg = False, metric = 'all', compute_scores = True, use_visual_feature_extractor = True, verbose = False, test_se = False, demo = False):


        #%%#######################################################################
        # Choosing the deep generative speech model
        ##########################################################################
        # - 'FFNN': FFNN generative speech model
        # - 'RNNenc-RNNdec': RNN generative speech model
        # - 'BRNNenc-BRNNdec': BRNN generative speech model
        ##########################################################################


        self.saved_model = saved_model

        self.save_flg = save_flg
        self.device = device
        self.output_dir = output_dir

        self.test_se = test_se
        self.demo = demo

        #%% Parameters

        self.compute_scores = compute_scores
        self.verbose = verbose

        path, fname = os.path.split(saved_model)
        self.config_file = os.path.join(path, 'config.ini')
        self.cfg = myconf()
        self.cfg.read(self.config_file)

        self.eps = np.finfo(float).eps # machine epsilon

        self.niter = niter 
        self.nmf_rank = nmf_rank

        self.vae_mode = self.cfg.get('Network', 'vae_mode')
        self.vae_type = self.cfg.get('Network', 'name')



        # Load STFT parameters
        self.wlen_sec = self.cfg.getfloat('STFT', 'wlen_sec')
        self.hop_percent = self.cfg.getfloat('STFT', 'hop_percent')
        self.fs = self.cfg.getint('STFT', 'fs')
        self.zp_percent = self.cfg.getint('STFT', 'zp_percent')
        self.wlen = self.wlen_sec * self.fs
        self.wlen = np.int(np.power(2, np.ceil(np.log2(self.wlen)))) # pwoer of 2
        self.hop = np.int(self.hop_percent * self.wlen)
        self.nfft = self.wlen + self.zp_percent * self.wlen
        self.win = np.sin(np.arange(0.5, self.wlen+0.5) / self.wlen * np.pi)

        # Load Visual Feature Extractor
        self.use_visual_feature_extractor = use_visual_feature_extractor
        if self.use_visual_feature_extractor:
            self.vfeats = self.build_visual_extractor()


        #%% Load VAE

        if self.vae_type == 'VAE':

            self.vae = build_VAE(cfg = self.cfg, device = self.device, vae_mode = self.vae_mode)

        elif self.vae_type == 'DKF':

            self.vae = build_DKF(cfg = self.cfg, device = self.device, vae_mode = self.vae_mode)

        else:
            raise NameError('Unknown VAE type')

        self.vae.load_state_dict(torch.load(self.saved_model, map_location= self.device), strict = True)
        self.vae.eval()

    def build_visual_extractor(self):
        if self.test_se:
            config_path = "../../lipreading/data/lrw_resnet18_mstcn.json"
        elif self.demo:
            config_path = "./lipreading/data/lrw_resnet18_mstcn.json"
        else:
            config_path = "lipreading/data/lrw_resnet18_mstcn.json"

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

        vfeats = Lipreading( modality='video',
                            num_classes=500,
                            tcn_options=tcn_options,
                            backbone_type=backbone_type,
                            relu_type=relu_type,
                            width_mult=width_mult,
                            extract_feats=True).to(self.device)
        if self.test_se:
            vfeats = load_model("../../lipreading/data/lrw_resnet18_mstcn_adamw_s3.pth.tar", vfeats, allow_size_mismatch=False)
        elif self.demo:
            vfeats = load_model("./lipreading/data/lrw_resnet18_mstcn_adamw_s3.pth.tar", vfeats, allow_size_mismatch=False)
        else:
            vfeats = load_model("lipreading/data/lrw_resnet18_mstcn_adamw_s3.pth.tar", vfeats, allow_size_mismatch=False)

        vfeats.eval()

        return vfeats

    def get_specs(self, path_mix, speech_file, path_n_hat, path_s_hat):
        y1, sr1 = librosa.load(path_mix)
        y2, sr2 = librosa.load(speech_file)
        y3, sr3 = librosa.load(path_n_hat)
        y4, sr4 = librosa.load(path_s_hat)


        D1 = librosa.amplitude_to_db(np.abs(librosa.stft(y1)), ref=np.max)
        D2 = librosa.amplitude_to_db(np.abs(librosa.stft(y2)), ref=np.max)
        D3 = librosa.amplitude_to_db(np.abs(librosa.stft(y3)), ref=np.max)
        D4 = librosa.amplitude_to_db(np.abs(librosa.stft(y4)), ref=np.max)

        spec_figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

        img1 = librosa.display.specshow(D1, y_axis='log', x_axis='time',

                                       sr=sr1, ax= ax1)
        # plt.show()
        img2 = librosa.display.specshow(D2, y_axis='log', x_axis='time',

                                       sr=sr2, ax= ax2)
        # plt.show()
        img3 = librosa.display.specshow(D3, y_axis='log', x_axis='time',

                                       sr=sr3, ax= ax3)
        # plt.show()
        img4 = librosa.display.specshow(D4, y_axis='log', x_axis='time',

                                       sr=sr4, ax= ax4)
        return spec_figure

    def run(self, input_files, tqdm=None, experience_name = None, compute_trace=False):

        if experience_name is None:
            self.experience_name = self.vae_mode
        else:
            self.experience_name = experience_name

        #%% Load input signals and compute STFT

        mix_file, speech_file, video_file, algo_type = input_files

        s_orig, fs = sf.read(speech_file) # clean speech
        x, fx = sf.read(mix_file)    # mixture speech


        s_orig = s_orig/np.max(s_orig)
        x = x/np.max(x)

        T_orig = len(x)


        X = librosa.stft(x, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen,
                         window=self.win)


        s_stft = librosa.stft(s_orig, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen,
                         window=self.win)
        F, N = X.shape
        X_abs_2 = np.abs(X)**2
        S_abs_2 = np.abs(s_stft)**2


        N_aframes = X.shape[1]

        ########################### upsample video ############################


        v_orig = np.load(video_file)  # video

        if not self.use_visual_feature_extractor:

            F, N = X.shape
            N_aframes = X.shape[1]

            ########################### upsample video ############################
            v = resample(v_orig, target_num = N_aframes)

            # print(v.shape)
            v = v.reshape([67,67, 1, -1])
            data_orig_v = torch.from_numpy(v.astype(np.float32)).to(self.device)
        else:
            N_vframes = v_orig.shape[1]
            N_aframes = X.shape[1]

            #%% Video resampling

            T_orig_2 = len(x)/fs
            fps = 30

            v_trimmed = v_orig
            data_v_to_save = v_orig.copy().transpose().reshape((-1,1,67,67,1))

            data_v = resample(v_trimmed, target_num = N_aframes)
            data_v = data_v.transpose().reshape((-1,1,67,67,1))
 
            this_dtype = torch.cuda.FloatTensor if self.device == 'cuda' else torch.FloatTensor
            data_v = torch.from_numpy(data_v).permute(1,-1,0,2,3).to(self.device).type(this_dtype)
            v = self.vfeats(data_v, lengths=None)[0,...].detach().cpu().numpy()
            data_orig_v = torch.from_numpy(v.astype(np.float32).transpose()).to(self.device)


        attention = False

        stft_param = {"hop": self.hop, "wlen": self.wlen, "win": self.win, "len": T_orig}


        if "peem" in algo_type or "fast" in algo_type or "mcem" in algo_type:

            # Compute the first latent variable sample
            with torch.no_grad():
                X_abs_2 = X_abs_2.T
                X_abs_2 = torch.from_numpy(X_abs_2.astype(np.float32))
                X_abs_2 = X_abs_2.to(self.device)
                if self.vae_type == 'DKF':
                    _,Z_init, _ = self.vae.inference(X_abs_2.unsqueeze(0).permute(1, 0, -1), data_orig_v.unsqueeze(0).permute(-1,0,1)) # corresponds to the encoder mean

                elif self.vae_type == 'VAE':
                    _,Z_init, _ = self.vae.inference(X_abs_2, data_orig_v.permute(1,0))
                Z_init = Z_init.cpu().numpy().T


        #%% Initialize noise model parameters
        np.random.seed(23)
        W_init = np.maximum(np.random.rand(F,self.nmf_rank), self.eps)
        np.random.seed(23)
        H_init = np.maximum(np.random.rand(self.nmf_rank, N), self.eps)

        g_init = np.ones(N)

        #%% Define speech enhancement algorithm
        #%%#######################################################################
        # Choosing the speech enhancement algorithm
        ##########################################################################
        # - 'mcem': Monte Carlo EM (only valid for FFNN generative speech model)
        # - 'vem': Variational EM with fine-tune encoder (valid for all models)
        # - 'peem': Point-estimate EM (valid for all models)
        ##########################################################################

        if algo_type == 'mcem':

            print('MCEM')

            nsamples_E_step = 10
            nsamples_WF = 25
            burnin_E_step = 30
            burnin_WF = 75
            var_RW = 0.01

            algo = MCEM(X=X, Vf = v, W=W_init, H=H_init, g=g_init, Z=Z_init, vae=self.vae,
                        device=self.device, niter=self.niter, nsamples_E_step=nsamples_E_step,
                        burnin_E_step=burnin_E_step, nsamples_WF=nsamples_WF,
                        burnin_WF=burnin_WF, var_RW=var_RW, verbose = self.verbose)

        elif algo_type == 'vem':

            print('VEM')

            nsamples_E_step = 1
            nsamples_WF = 1
            lr = 1e-2

            if self.vae_type == 'VAE':
                nepochs_E_step = 10 # IMPORTANT PARAMETER: If 1 epoch, bad results
            else:
                nepochs_E_step = 1

            algo = VEM(X=X, Vf = v, W=W_init, H=H_init, g=g_init, vae=self.vae, device=self.device,
                       niter=self.niter, nsamples_E_step=nsamples_E_step,
                       nsamples_WF=nsamples_WF, lr=lr,
                       nepochs_E_step=nepochs_E_step, verbose = self.verbose)

        elif algo_type == 'dvem':

            print('DVEM')

            nsamples_E_step = 1
            nsamples_WF = 1
            lr = 1e-3
            nepochs_E_step = 20

            algo = DVEM(X=X, Vf = v.transpose(), W=W_init, H=H_init, g=g_init, dvae = self.vae, dvae_type = 'DKF',
                          niter=self.niter, lr=lr, nepochs_E_step=nepochs_E_step,
                          nsamples_WF=nsamples_WF, device=self.device)

        elif algo_type == 'peem':

            print('PEEM')

            lr = 1e-2

            nepochs_E_step = 20

            algo = PEEM(X=X, Vf = v, W=W_init, H=H_init, g=g_init, Z=Z_init, vae=self.vae,
                        device=self.device, niter=self.niter, lr=lr,
                        nepochs_E_step=nepochs_E_step, verbose = self.verbose)

        elif algo_type == 'gpeem':

            print('PEEM')

            lr = 1e-2

            nepochs_E_step = 20

            algo = GPEEM(X=X, Vf = v, W=W_init, H=H_init, g=g_init, Z=Z_init, vae=self.vae,
                        device=self.device, niter=self.niter, lr=lr,
                        nepochs_E_step=nepochs_E_step, attention = attention, verbose = self.verbose)

        elif algo_type == 'dpeem':

            print('DPEEM')

            lr = 3e-3

            nepochs_E_step = 20

            algo = DPEEM(X=X, Vf = v.T, W=W_init, H=H_init, g=g_init, Z=Z_init, vae=self.vae,
                        device=self.device, niter=self.niter, lr=lr,
                        nepochs_E_step=nepochs_E_step, attention = attention, verbose = self.verbose)
        elif algo_type == 'gdpeem':

            print('GDPEEM')

            lr = 3e-3

            nepochs_E_step = 20

            algo = GDPEEM(X=X, Vf = v, W=W_init, H=H_init, g=g_init, Z=Z_init, vae=self.vae, visual = data_orig_v.unsqueeze(0).permute(-1,0,1),
                        device=self.device, niter=self.niter, lr=lr,
                        nepochs_E_step=nepochs_E_step, attention = attention, verbose = self.verbose, alpha = 0.0, Z_oracle = Z_init, is_z_oracle = False, is_noise_oracle = False, fix_gain = True, rec_power = 0.9)


        else:

            raise NameError('Unknown algorithm')

        #%% Run speech enhancement algorithm
        if compute_trace:
            spec_param = {"mix": X, "clean": s_stft}
            algo_run_params = {"tqdm": tqdm, "stft_param": stft_param, "eval_metric": self.eval_metrics, "s_orig": s_orig, "spec_param":spec_param, "sample_rate": fx}
        else:
            algo_run_params = {"tqdm": tqdm}
        algo.run(params = algo_run_params)

        #%% Save estimated sources
        path0, mix_name = os.path.split(mix_file)
        path01, _ = os.path.split(path0)
        path1, speaker_id = os.path.split(path01)
        path11, _ = os.path.split(path1)
        path2, snr_level = os.path.split(path11)
        _, noise_type = os.path.split(path2)

        save_dir = os.path.join(self.output_dir, noise_type, str(snr_level), speaker_id)

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        s_hat = librosa.istft(stft_matrix=algo.S_hat, hop_length=self.hop,
                              win_length=self.wlen, window=self.win, length=T_orig)
        n_hat = librosa.istft(stft_matrix=algo.N_hat, hop_length=self.hop,
                              win_length=self.wlen, window=self.win, length=T_orig)



        metrics_dict = get_metrics(mix = x, clean = s_orig, estimate = s_hat, sample_rate=fs, metrics_list=['si_sdr', 'stoi', 'pesq'])
        
        input_scores = [metrics_dict['input_si_sdr'], metrics_dict['input_pesq'], metrics_dict['input_stoi']]


        #%% Evaluation
        
        output_scores = []
        
        if self.compute_scores:
            info = {"input_scores": input_scores}
            list_score = [metrics_dict['si_sdr'], metrics_dict['pesq'], metrics_dict['stoi'], info]
            output_scores = [metrics_dict['si_sdr'], metrics_dict['pesq'], metrics_dict['stoi']]
            info = {"output_scores": output_scores}
            list_sample = {"speaker_id": speaker_id,
                                 "noise_type": noise_type,
                                 "SNR": snr_level,
                                 "mix_file": mix_file,
                                 "speech_file": speech_file,
                                 "video_file": video_file}

            if self.save_flg:
                np.savez(os.path.join(save_dir, noise_type + '_' + str(snr_level) + '_' + speaker_id + '_' + mix_name[:-4]+'.npz'), list_score = list_score, list_sample = list_sample,
                        input_scores = input_scores)

        spec_figure = None
        if self.save_flg:
            out_path = os.path.join(save_dir,mix_name[:-4]+'_video.avi')
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps,(67,67))
            for idx in tqdm(range(data_v_to_save.shape[0])):
                frame = cv2.cvtColor(np.uint8(data_v_to_save[idx, 0, :, :, 0].transpose()*255),cv2.COLOR_GRAY2BGR)
                out.write(frame)
            out.release()
            cv2.destroyAllWindows()
            path_s_hat = os.path.join(save_dir, mix_name[:-4]+f'_speech_est_{algo_type}_{self.experience_name}' +'.wav')
            sf.write(path_s_hat, s_hat, self.fs)
            path_n_hat = os.path.join(save_dir,mix_name[:-4]+f'_noise_est_{algo_type}_{self.experience_name}' +'.wav')
            sf.write(path_n_hat, n_hat, self.fs)
            path_mix = os.path.join(save_dir,mix_name[:-4]+'_mix_norm.wav')
            sf.write(path_mix, x, self.fs)
            spec_figure = self.get_specs(path_mix, speech_file, path_n_hat, path_s_hat)

        info = {
            "spec_figure": spec_figure,
            "input_scores": input_scores,
            "output_scores": output_scores,
            "S_hat_wave": s_hat,
            "N_hat_wave": n_hat,
            "S_hat_spec": algo.S_hat,
            "noisy_wave": x,
            "noisy_spec": X,
            "clean_wave": s_orig,
            "clean_spec": s_stft
        }
        return info
