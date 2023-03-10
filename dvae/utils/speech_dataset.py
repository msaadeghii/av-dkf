#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt

Class SpeechSequencesFull():
- generate Pytorch dataloader
- data sequence is clipped from the beginning of each audio signal
- every speech sequence can be divided into multiple data sequences, as long as audio_len >= seq_len
- usually, this method will give larger training sequences

Class SpeechSequencesRandom():
- generate Pytorch dataloader
- data sequence is clipped from a random place in each audio signal
- every speech sequence can only be divided into one single data sequence
- this method will introduce some randomness into training dataset

Both of these two Class use librosa.effects.trim()) to trim leading and trailing silence from an audio signal

Adapted and modified by Mostafa Sadeghi (mostafa.sadeghi@inria.fr) and Ali Golmakani (golmakani77@yahoo.com).

"""

import os
import numpy as np
import soundfile as sf
import librosa
import random
import torch
from torch.utils import data

from lipreading.utils import load_json, save2npz
from lipreading.utils import showLR, calculateNorm2, AverageMeter
from lipreading.utils import load_model, CheckpointSaver
from lipreading.model import Lipreading

from tqdm import tqdm
import random

import torchvision.transforms as T
from PIL import Image

def resample(video, target_num):
    n, N = video.shape # (4489, 129)
    ratio = N / target_num
    idx_lst = np.arange(target_num).astype(float)
    idx_lst *= ratio
    res = np.zeros((n, target_num))
    for i in range(target_num):
        res[:,i] = video[:,int(idx_lst[i])]
    return res


class SpeechDatasetFrames(data.Dataset):
    """
    Customize a dataset of speech sequences for Pytorch
    at least the three following functions should be defined.

    This is a quick speech sequence data loader which allow multiple workers
    """
    def __init__(self, file_list, STFT_dict, shuffle, sequence_len = 'NaN', name='WSJ0', trim=True, extract_visual_features=False, vf_root = None):

        super().__init__()

        # STFT parameters
        self.STFT_dict = STFT_dict
        self.fs = STFT_dict['fs']
        self.nfft = STFT_dict['nfft']
        self.hop = STFT_dict['hop']
        self.wlen = STFT_dict['wlen']
        self.win = STFT_dict['win']
        self.trim = STFT_dict['trim']

        # data parameters
        self.file_list = file_list
        self.sequence_len = sequence_len
        self.name = name
        self.shuffle_file_list = shuffle
        self.current_frame = 0
        self.tot_num_frame = 0
        self.cpt_file = 0
        self.trim = trim
        self.extract_visual_features = extract_visual_features

        self.compute_len()
        self.vf_root = vf_root if vf_root is not None else "vf"

    def compute_len(self):

        self.num_samples = 0

        for cpt_file, wavfile in enumerate(self.file_list):

            x, fs_x = sf.read(wavfile)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')

            # remove beginning and ending silence
            x, index = librosa.effects.trim(x, top_db=30)

            x = np.pad(x, int(self.nfft // 2), mode='reflect')
            # (cf. librosa.core.stft)

            n_frames = 1 + int((len(x) - self.wlen) / self.hop)

            self.num_samples += n_frames



    def __len__(self):
        """
        arguments should not be modified
        Return the total number of samples
        """
        return self.num_samples


    def __getitem__(self, index):
        """
        input arguments should not be modified
        torch data loader will use this function to read ONE sample of data
        from a list that can be indexed by parameter 'index'
        """

        if self.current_frame == self.tot_num_frame:

            if self.cpt_file == len(self.file_list):
                self.cpt_file = 0
                if self.shuffle_file_list:
                    random.shuffle(self.file_list)

            wavfile = self.file_list[self.cpt_file]
            self.cpt_file += 1

            # Data Audio
            x, fs_x = sf.read(wavfile)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')

            x_orig = x/np.max(np.abs(x))

            # remove beginning and ending silence
            if self.trim:
                x_trimmed, index = librosa.effects.trim(x_orig, top_db=30)
            else:
                x_trimmed = x_orig

            T_orig = len(x_trimmed)

            X = librosa.stft(x_trimmed, n_fft=self.STFT_dict["nfft"], hop_length=self.STFT_dict["hop"],
                             win_length=self.STFT_dict["wlen"],
                             window=self.STFT_dict["win"]) # STFT
            self.data_a = np.abs(X)**2 # shape (F, N)

            # Data Video
            path, fname = os.path.split(wavfile)
            
            videofile = os.path.join(path, fname[:-4]+'Raw.npy')

            v_orig = np.load(videofile)
            N_vframes = v_orig.shape[1]
            N_aframes = X.shape[1]

            #%% Video resampling

            if self.trim:
                t1, t2 = index/self.STFT_dict["fs"]
            else:
                t1 = 0
                t2 = T_orig

            T_orig = len(x_orig)/self.STFT_dict["fs"]
            fps = 30

            v1, v2 = int(np.floor(t1*fps)), int(np.floor((T_orig-t2)*fps))

            v_trimmed = v_orig[:,v1:N_vframes-v2]

            N_aframes = X.shape[1]

            v_up = resample(v_trimmed, target_num = N_aframes)


            self.data_v = v_up # shape (F, N)


            self.current_frame = 0
            self.tot_num_frame = self.data_a.shape[1]

        frame_a = self.data_a[:,self.current_frame]

        if not self.extract_visual_features:
            frame_v = np.asarray([self.data_v[:,i].reshape([67,67, 1]) for i in [max(0, t) for t in range(self.current_frame-5, self.current_frame)]])
        else:
            frame_v = self.data_v[:,self.current_frame]

        self.current_frame += 1

        frame_a = torch.from_numpy(frame_a.astype(np.float32))
        frame_v = torch.from_numpy(frame_v.astype(np.float32))

        return frame_a, frame_v


class SpeechSequencesFull(data.Dataset):
    """
    Customize a dataset of speech sequences for Pytorch
    at least the three following functions should be defined.
    """
    def __init__(self, file_list, sequence_len, overlap, STFT_dict, shuffle, name='WSJ0', extract_visual_features=False, gaussian = 0, vf_root = None):

        super().__init__()

        # STFT parameters
        self.STFT_dict = STFT_dict
        self.fs = STFT_dict['fs']
        self.nfft = STFT_dict['nfft']
        self.hop = STFT_dict['hop']
        self.wlen = STFT_dict['wlen']
        self.win = STFT_dict['win']
        self.trim = STFT_dict['trim']
        self.gaussian = gaussian
        # data parameters
        self.file_list = file_list
        self.sequence_len = sequence_len
        self.name = name
        self.shuffle = shuffle
        self.overlap = overlap
        self.extract_visual_features = extract_visual_features
        self.compute_len()
        self.vf_root = vf_root if vf_root is not None else "vf"
        self.preprocess = T.Compose([
                                        T.RandomHorizontalFlip(p=0.5),
                                        T.ColorJitter(brightness=0.4, contrast=0.4),
                                        T.RandomAffine(10, translate=(0.1, 0.1), scale=(0.9,1.1)),
                                        T.Resize((88,88)),
                                        T.ToTensor(),
                                        T.Normalize(
                                            mean=[0.421],
                                            std=[0.165]
                                        )
                                    ])

    def compute_len(self):

        self.valid_seq_list = []

        for wavfile in self.file_list:

            x, fs_x = sf.read(wavfile)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')

            # remove beginning and ending silence
            if self.trim:
                x_t, (ind_beg, ind_end) = librosa.effects.trim(x, top_db=30)
            else:
                x_t = x
            x = np.pad(x_t, int(self.nfft // 2), mode='reflect')
            # (cf. librosa.core.stft)
            X = librosa.stft(x, n_fft=self.nfft, hop_length=self.hop,
                             win_length=self.wlen,
                             window=self.win) # STFT
            # Check valid wav files

            file_length = X.shape[1]
            n_seq = int(((int(file_length - self.sequence_len)) // self.sequence_len)//self.overlap)
            for i in range(n_seq):
                seq_start = int(i * self.sequence_len*self.overlap)
                seq_end = int(i * self.sequence_len*self.overlap) + self.sequence_len
                seq_info = (wavfile, seq_start, seq_end)
                self.valid_seq_list.append(seq_info)


        if self.shuffle:
            random.shuffle(self.valid_seq_list)


    def __len__(self):
        """
        arguments should not be modified
        Return the total number of samples
        """
        return len(self.valid_seq_list)


    def __getitem__(self, index):
        """
        input arguments should not be modified
        torch data loader will use this function to read ONE sample of data from a list that can be indexed by
        parameter 'index'
        """

        # Read wav files
        wavfile, seq_start, seq_end = self.valid_seq_list[index]
        x_orig, fs_x = sf.read(wavfile)

        if self.trim:
            x, index_trim = librosa.effects.trim(x_orig, top_db=30)
        else:
            x = x_orig
        x = np.pad(x, int(self.nfft // 2), mode='reflect')

        # Normalize sequence
        x = x/np.max(np.abs(x))
        
        # Add small Gaussian noise
        if self.gaussian > 0:
                x += self.gaussian * np.random.randn(*x.shape)

        # STFT transformation
        audio_spec = torch.stft(torch.from_numpy(x), n_fft=self.nfft, hop_length=self.hop,
                                win_length=self.wlen, window=torch.from_numpy(self.win),
                                center=True, pad_mode='reflect', normalized=False, onesided=True, return_complex = False)


        # Square of magnitude
        data_a = (audio_spec[:,:,0]**2 + audio_spec[:,:,1]**2).float()
        phase_a = torch.angle(audio_spec[:,:,0].squeeze() + 1j * audio_spec[:,:,1].squeeze())
        
        sample_a = data_a[...,seq_start:seq_end]
        sample_phase_a = phase_a[...,seq_start:seq_end]
        
        # Read video file
        path, fname = os.path.split(wavfile)

        videofile = os.path.join(path, fname[:-4]+'RawVF.npy') # (512, N_frames)
        v_orig = np.load(videofile)

        N_vframes = v_orig.shape[1]
        N_aframes = data_a.shape[1]

        #%% Video resampling
        T_orig = len(x_orig)/self.STFT_dict["fs"]
        if self.trim:
                t1, t2 = index_trim/self.STFT_dict["fs"]
        else:
            t1 = 0
            t2 = T_orig

        fps = 30

        v1, v2 = int(np.floor(t1*fps)), int(np.floor((T_orig-t2)*fps))

        v_trimmed = v_orig[:,v1:N_vframes-v2]

        data_v = resample(v_trimmed, target_num = N_aframes).T

        sample_v = data_v.T
        sample_v = sample_v[...,seq_start:seq_end]

        return sample_a, sample_v, sample_phase_a


class SpeechDatasetSequences(data.Dataset):
    """
    Customize a dataset for PyTorch, in order to be used with torch
    dataloarder, at least the three following functions should be defined.
    """

    def __init__(self, file_list, sequence_len, overlap, STFT_dict, shuffle, name='WSJ0', extract_visual_features=False):

        super().__init__()

        self.file_list = file_list

        # STFT parameters
        self.fs = STFT_dict['fs']
        self.nfft = STFT_dict['nfft']
        self.hop = STFT_dict['hop']
        self.wlen = STFT_dict['wlen']
        self.win = STFT_dict['win']
        self.trim = STFT_dict['trim']

        self.name = name

        self.cpt_file = 0
        self.current_frame = 0
        self.tot_num_frame = 0
        self.data = None
        self.shuffle_file_list = shuffle

        self.sequence_len = sequence_len
        self.overlap = int(overlap*sequence_len)
        self.compute_len()
        self.is_extract_visual_features = extract_visual_features


        if extract_visual_features:
            config_path = "./lipreading/data/lrw_resnet18_mstcn.json"
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
            self.vfeats = load_model("./lipreading/data/lrw_resnet18_mstcn_adamw_s3.pth.tar", self.vfeats, allow_size_mismatch=False)
            self.vfeats.eval()
            self.extract_visual_features()

    def compute_len(self):

        self.num_samples = 0

        for cpt_file, wavfile in enumerate(self.file_list):


            x, fs_x = sf.read(wavfile)
            if self.fs != fs_x:
                raise ValueError('Unexpected sampling rate')

            # remove beginning and ending silence
            x, index = librosa.effects.trim(x, top_db=30)

            x = np.pad(x, int(self.nfft // 2), mode='reflect')
            # (cf. librosa.core.stft)

            n_seq = (1 + int((len(x) - self.wlen) / self.hop) )//self.overlap
            # n_seq can be equal to 0 if some audio files are too short
            # compared with the expected sequence length

            self.num_samples += n_seq

    def __len__(self):
        """
        arguments should not be modified
        Return the total number of samples
        """
        return self.num_samples

    def extract_visual_features(self):
        for wavfile in tqdm(self.file_list):

            path, fname = os.path.split(wavfile)
            videofile = os.path.join(path, fname[:-4]+'Raw.npy')
            v_orig = np.load(videofile)

            data_v = v_orig.transpose().reshape((-1,1,67,67,1)).copy()
            data_v_hf = np.flip(v_orig.transpose().reshape((-1,1,67,67,1)).copy(), axis = 2) # Horizontal Flip
            
            data_v = torch.from_numpy(data_v.astype(np.float32).copy()).permute(1,-1,0,2,3).to('cuda')
            data_v_hf = torch.from_numpy(data_v_hf.astype(np.float32).copy()).permute(1,-1,0,2,3).to('cuda')
            
            data_v = self.vfeats(data_v, lengths=None)[0,...].float().detach().cpu().numpy()
            data_v_hf = self.vfeats(data_v_hf, lengths=None)[0,...].float().detach().cpu().numpy()
            
            path, fname = os.path.split(videofile)
            newpath = os.path.join(path, fname[:-4]+"VF.npy")
            np.save(newpath, data_v.T)
            
            newpath = os.path.join(path, fname[:-4]+"VF_hf.npy")
            np.save(newpath, data_v_hf.T)


    def __getitem__(self, index):
        """
        input arguments should not be modified
        torch data loader will use this function to read ONE sample of data
        from a list that can be indexed by parameter 'index'
        """

        if  (self.tot_num_frame - self.current_frame) < self.sequence_len:

            while True:

                if self.cpt_file == len(self.file_list):
                    self.cpt_file = 0
                    if self.shuffle_file_list:
                        random.shuffle(self.file_list)

                wavfile = self.file_list[self.cpt_file]
                self.cpt_file += 1


                x, fs_x = sf.read(wavfile)
                if self.fs != fs_x:
                    raise ValueError('Unexpected sampling rate')
                x_orig = x/np.max(np.abs(x))

                # remove beginning and ending silence
                x, index_ = librosa.effects.trim(x_orig, top_db=30)

                x = np.pad(x, int(self.nfft // 2), mode='reflect')
                # (cf. librosa.core.stft)
                X = librosa.stft(x, n_fft=self.nfft, hop_length=self.hop,
                                 win_length=self.wlen,
                                 window=self.win) # STFT

                self.data_a = np.abs(X)**2
                self.current_frame = 0
                self.tot_num_frame = self.data_a.shape[1]

                # Read video file
                path, fname = os.path.split(wavfile)
                if self.is_extract_visual_features:
                    self.data_v = np.load(os.path.join("vf", path[-10:].replace("/",'_')+fname[:-4]+"VF.npy")).transpose()
                    # print("cool")
                    # print(self.data_v.shape)
                    # mozz
                else:
                    videofile = os.path.join(path, fname[:-4]+'Raw.npy')
                    v_orig = np.load(videofile)


                    N_vframes = v_orig.shape[1]
                    N_aframes = self.data_a.shape[1]

                    #%% Video resampling

                    t1, t2 = index_/self.fs
                    T_orig = len(x_orig)/self.fs
                    fps = 30

                    v1, v2 = int(np.floor(t1*fps)), int(np.floor((T_orig-t2)*fps))

                    v_trimmed = v_orig[:,v1:N_vframes-v2]

                    self.data_v = resample(v_trimmed, target_num = N_aframes)


                if self.tot_num_frame >= self.sequence_len:
                    break

        sample_a = self.data_a[:,self.current_frame:self.current_frame + self.sequence_len]


        if not self.is_extract_visual_features:
            sample_v = self.data_v[:,self.current_frame:self.current_frame + self.sequence_len].reshape([67,67, 1, self.sequence_len])
        else:
            sample_v = self.data_v[:,self.current_frame:self.current_frame + self.sequence_len]


        if sample_a.shape[1] != self.sequence_len:
            print(self.data_a.shape)


        self.current_frame += self.overlap

        # turn numpy array to torch tensor with torch.from_numpy#

        """
        e.g.
        matrix = torch.from_numpy(matrix.astype(np.float32))
        target = torch.from_numpy(np.load(t_pth).astype(np.int32))
        """

        sample_a = torch.from_numpy(sample_a.astype(np.float32))
        sample_v = torch.from_numpy(sample_v.astype(np.float32))

        return sample_a, sample_v
