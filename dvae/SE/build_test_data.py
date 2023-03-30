#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configure the test data of NTCD-TIMIT for speech enhancement evaluation.

@author: Mostafa Sadeghi (mostafa.sadeghi@inria.fr)
"""


import os
import sys
from six.moves import cPickle as pickle #for performance

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)
 
def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def get_data(speaker_id, noise_type, SNR):

    fnames = [x[:-4] for x in os.listdir(os.path.join(mix_dir, noise_type, str(SNR), 'volunteers', speaker_id, 'straightcam')) if x.endswith(".wav")]

    out_dict = {fname: [os.path.join(mix_dir, noise_type, str(SNR), 'volunteers', speaker_id, 'straightcam', fname+'.wav'),
                os.path.join(speech_dir, speaker_id, fname+'.wav'),
                os.path.join(speech_dir, speaker_id, fname+'Raw.npy')] for fname in  fnames}

    return fnames, out_dict

file_limit = 5 # maximum number of test files per speaker
data_dir = '/srv/storage/talc@storage4.nancy.grid5000.fr/multispeech/corpus/audio_visual'
save_dir = os.path.join(data_dir, 'TCD-TIMIT', 'test_data_NTCD')

speech_dir = os.path.join(data_dir, 'TCD-TIMIT', 'test_data_NTCD', 'clean')
mix_dir = os.path.join(data_dir, 'NTCD-TIMIT-noisy')

speakers_id = ['09F',  '24M',  '26M',  '27M',  '33F',  '40F',  '47M',  '49F', '56M']
noise_types = ['Babble',  'Cafe',  'Car',  'LR', 'Street', 'White']
SNRs = [-10,-5,0,5,10]

# Iterate through all the test speech files:
files_list = []

for sid in speakers_id:
    for nt in noise_types:
        for snr in SNRs:
            fnames, out_dict = get_data(sid, nt, snr)
            fnames.sort()
            fnames = fnames[:file_limit]

            for fname in fnames:

                mix_file, speech_file, video_file = out_dict[fname]
                
                this_data = {'speech_file':speech_file,
                             'mix_file':mix_file,
                             'video_file':video_file,
                             'noise_type': nt,
                             'snr':snr,
                             'speaker_id':sid,
                             'file_name':fname}
                
                files_list.append(this_data)

data_info = {'file_limit':file_limit,
             'speakers_id':speakers_id,
             'noise_types':noise_types,
             'SNRs':SNRs}

if not os.path.exists(os.path.join(save_dir, f'test_data_{file_limit}.pkl')):
    save_dict(files_list, os.path.join(save_dir, f'test_data_{file_limit}.pkl'))
    save_dict(data_info, os.path.join(save_dir, f'test_data_{file_limit}_info.pkl'))
    