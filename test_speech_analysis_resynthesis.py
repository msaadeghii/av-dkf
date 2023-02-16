#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 15:35:08 2021

This code runs speech analysis-resynthesis on the test set of TCD-TIMIT.

@author: Mostafa Sadeghi (mostafa.sadeghi@inria.fr) and Ali Golmakani (golmakani77@yahoo.com).

"""


from dvae import LearningAlgorithm
import os
from random import shuffle
import numpy as np
from tqdm import tqdm
import sys


def evaluate(model_path, num_samples = 1000):

    root_dir = '/pathTo/corpus/'

    speech_dir = os.path.join(root_dir, 'audio_visual', 'TCD-TIMIT', 'test_data_NTCD', 'clean')

    file_list = [os.path.join(root, name)
              for root, dirs, files in os.walk(speech_dir)
              for name in files
              if name.endswith('.wav')]

    #shuffle(file_list)

    file_list = file_list[0:num_samples]
    
    # Create score list
    list_score_isdr = []
    list_score_pesq= []
    list_score_stoi = []

    list_spm = []

    save_dir = './resynthesises'
    save_flag = False
    save_video = False
    trim = True

    state_dict_file = model_path


    path, fname = os.path.split(state_dict_file)
    cfg_file = os.path.join(path, 'config.ini')
    learning_algo = LearningAlgorithm(config_file=cfg_file)


    for ind_mix, speech_file in tqdm(enumerate(file_list)):

        audio_recon = os.path.join(save_dir, f'a-dkf-normal-{ind_mix}.wav')

        if "VAE" in learning_algo.vae_mode:
            score_isdr, score_pesq, score_stoi, spm = learning_algo.generate(audio_orig = speech_file, audio_recon = audio_recon, save_flag = save_flag,  state_dict_file = state_dict_file, seed = ind_mix, model_type = learning_algo.vae_mode, save_video = save_video, trim_s = trim)
        else:
            score_isdr, score_pesq, score_stoi, spm = learning_algo.generate_dkf(audio_orig = speech_file, audio_recon = audio_recon, save_flag = save_flag,  state_dict_file = state_dict_file)

        list_score_isdr.append(score_isdr)
        list_score_pesq.append(score_pesq)
        list_score_stoi.append(score_stoi)
        list_spm.append(spm)

    print('SDR = {} '.format(np.mean(np.asarray(list_score_isdr))))
    print('PESQ = {} '.format(np.mean(np.asarray(list_score_pesq))))
    print('STOI = {} '.format(np.mean(np.asarray(list_score_stoi))))
    print('SPM = {} '.format(np.mean(np.asarray(list_spm))))


if __name__ == '__main__':
    
    if len(sys.argv) == 2:
        path = sys.argv[1]
        evaluate(path)

    else:
        print('Error: Please input the config file')
