#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 20 12:09:03 2021

This code is to test the speech enhancement algorithms on the test set of the NTCD-TIMIT dataset.

@author: Mostafa Sadeghi (mostafa.sadeghi@inria.fr) and Ali Golmakani (golmakani77@yahoo.com).

"""

from speech_enhancement import SpeechEnhancement
import os
import torch
import sys

vae_mode = "A-DKF" # VAE model. Non-dynamical: A-VAE, AV-VAE. Dynamical: A-DKF, AV-DKF.
algo_type = "dpeem" # SE algorithm. Choose one of {peem, gpeem} for non-dynamical VAE models, and one of {dpeem, gdpeem} for the dynamical versions.

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if vae_mode == 'A-VAE':
    model_path = '../../pretrained_models/A-VAE/A-VAE.pt'
    cfg_file = '../../pretrained_models/A-VAE/config.ini'

if vae_mode == 'AV-VAE':
    model_path = '../../pretrained_models/AV-VAE/AV-VAE.pt'
    cfg_file = '../../pretrained_models/AV-VAE/config.ini'

if vae_mode == 'A-DKF':
    model_path = '../../pretrained_models/A-DKF/A-DKF.pt'
    cfg_file = '../../pretrained_models/A-DKF/config.ini'

if vae_mode == 'AV-DKF':
    model_path = '../../pretrained_models/AV-DKF/AV-DKF.pt'
    cfg_file = '../../pretrained_models/AV-DKF/config.ini'


path_model, _ = os.path.split(saved_model)
_, model_name = os.path.split(path_model)

output_dir = "./results"
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

verbose = True
test_se = True
save_flg = False
num_iter = 100 # number of EM iterations
num_E_step = 20 # number of E-step iterations
fs = 16000 # sampling frequency
nmf_rank = 8 # rank of the NMF model for noise
compute_scores = True # whether to compute the scores

# SE parameters
se_params = { 'model_path':model_path,
              'device':device,
              'output_dir':output_dir,
              'nmf_rank':nmf_rank,
              'num_iter':num_iter,
              'num_E_step':num_E_step,
              'save_flg':save_flg,
              'verbose':verbose,
              'test_se':test_se,
              'compute_scores':compute_scores}

se = SpeechEnhancement(se_params)


#%% Input signals

speakers_id = ['09F',  '24M',  '26M',  '27M',  '33F',  '40F',  '47M',  '49F', '56M']
noise_types = ['Babble',  'Cafe',  'Car',  'LR', 'Street', 'White']
SNRs = [-10,-5,0,5,10]

fs = 16000

root_dir = '/pathTo/corpus/'

speech_dir = os.path.join(root_dir, 'audio_visual', 'TCD-TIMIT', 'test_data_NTCD', 'clean')
mix_dir = os.path.join(root_dir, 'audio_visual', 'NTCD-TIMIT-noisy')

speaker_id = 3
noise_type = 3
snr_id = 1
fname = 'sa1'

mix_file = os.path.join(mix_dir, noise_types[noise_type], str(SNRs[snr_id]), 'volunteers', speakers_id[speaker_id], 'straightcam', fname+'.wav')
speech_file = os.path.join(speech_dir, speakers_id[speaker_id], fname+'.wav')
video_file = os.path.join(speech_dir, speakers_id[speaker_id], fname+'Raw.npy')

#%% Run SE & evaluations

info = se.run([mix_file, speech_file, video_file, algo_type])

print('Input scores - SI-SDR: {} -- PESQ: {} --- STOI: {}'.format(info['input_scores'][0], info['input_scores'][1], info['input_scores'][2]))
print('Output scores - SI-SDR: {} -- PESQ: {} --- STOI: {}'.format(info['output_scores'][0], info['output_scores'][1], info['output_scores'][2]))
