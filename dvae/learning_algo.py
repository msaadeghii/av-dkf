#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Software dvae-speech
Copyright Inria
Year 2020
Contact : xiaoyu.bie@inria.fr
License agreement in LICENSE.txt

The main python file for model training, data test and performance evaluation, see README.md for further information
"""

from comet_ml import Experiment
import pyloudnorm as pyln
from dvae.SE.speech_enhancement import SpeechEnhancement
from multiprocessing import Pool

import os
import shutil
import socket
import datetime
import numpy as np
import torch
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from .utils import myconf, get_logger, SpeechDatasetSequences, SpeechDatasetFrames, SpeechSequencesFull
from .model import build_VAE, build_DKF
from sklearn.metrics import mean_squared_error as mse_sk
import cv2
from tqdm import tqdm
import random
from asteroid.metrics import get_metrics

from .utils import cometml_logger
from lipreading.utils import load_json, save2npz
from lipreading.utils import showLR, calculateNorm2, AverageMeter
from lipreading.utils import load_model, CheckpointSaver
from lipreading.model import Lipreading

import configparser
from pprint import pprint

import pickle

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

class IterMeter(object):
    """keeps track of total iterations"""
    def __init__(self):
        self.val = 0

    def step(self):
        self.val += 1

    def get(self):
        return self.val
    
def ini_to_dict(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)
    
    config_dict = {}
    for section in config.sections():
        config_dict[section] = {}
        for key, value in config.items(section):
            config_dict[section][key] = value
    
    return config_dict

def Hoyer_sparsity(z):

    D = len(z)

    hoyer_sp = (np.sqrt(D) - np.linalg.norm(z, 1)/ np.linalg.norm(z, 2))/(np.sqrt(D) - 1)

    return hoyer_sp

def sparsity_measure(z):
    spm = 0
    N = z.shape[0]
    for i in range(N):
        spm += Hoyer_sparsity(z[i,:])

    return spm / N

def compute_rmse(x_est, x_ref):

    # align
    len_x = len(x_est)
    x_ref = x_ref[:len_x]
    # scaling
    alpha = np.sum(x_est*x_ref) / np.sum(x_est**2)
    # x_est_ = np.expand_dims(x_est, axis=1)
    # alpha = np.linalg.lstsq(x_est_, x_ref, rcond=None)[0][0]
    x_est_scaled = alpha * x_est
    return np.sqrt(np.square(x_est_scaled - x_ref).mean())

def resample(video, target_num):
    n, N = video.shape # (4489, 129)
    ratio = N / target_num
    idx_lst = np.arange(target_num).astype(float)
    idx_lst *= ratio
    res = np.zeros((n, target_num))
    for i in range(target_num):
        res[:,i] = video[:,int(idx_lst[i])]
    return res

def extract_frames(x, v_orig, STFT_dict, trim=True, extract_visual_features = True, fs = None, vfeats = None):
    x_orig = x/np.max(np.abs(x))

    # remove beginning and ending silence
    if trim:
        x_trimmed, index = librosa.effects.trim(x_orig, top_db=30)
    else:
        x_trimmed = x_orig

    T_orig = len(x_trimmed)

    X = librosa.stft(x_trimmed, n_fft=STFT_dict["nfft"], hop_length=STFT_dict["hop"],
                     win_length=STFT_dict["wlen"],
                     window=STFT_dict["win"]) # STFT

    if not extract_visual_features:

        F, N = X.shape
        N_aframes = X.shape[1]

        ########################### upsample video ############################
        v = resample(v_orig, target_num = N_aframes)

        v = v.transpose().reshape([-1, 67,67, 1])
        v_up = np.asarray([np.asarray([v[x, :, :, :] for x in [max(0, i) for i in range(k-5, k)]]) for k in range(v.shape[0])])
    else:
        N_vframes = v_orig.shape[1]
        N_aframes = X.shape[1]

        #%% Video resampling

        t1 = 1
        t2 = -1
        T_orig_2 = len(x)/fs
        fps = 30

        v1, v2 = int(np.floor(t1*fps)), int(np.floor((T_orig_2-t2)*fps))

        v_trimmed = v_orig[:,v1:N_vframes-v2]

        data_v = resample(v_trimmed, target_num = N_aframes)
        data_v = data_v.transpose().reshape((-1,1,67,67,1))
        data_v = torch.from_numpy(data_v).permute(1,-1,0,2,3).to('cuda').type(torch.cuda.FloatTensor)
        v_up = vfeats(data_v, lengths=None)[0,...].detach().cpu().numpy()


    return X, v_up, T_orig

class SE_Eval():
    """
    Class for evaluating the speech enhancement performance of a model during training.
    """
    
    def __init__(self, data_dir, demand_noise_dir, save_dir_eval,
                device = 'cuda', nmf_rank = 8, num_iter = 10, num_E_step = 5, algo_type = "gdpeem"):
        
        self.speech_dir = os.path.join(data_dir, 'val_data_NTCD')
        self.speakers_id = ['06M', '17F', '20M', '35M', '38F', '42M', '52M', '59F']
        self.noise_types = ['NPARK',  'DWASHING',  'PSTATION', 'TMETRO', 'DKITCHEN']
        self.SNRs = [0,5,10] 
        self.demand_noise_dir = demand_noise_dir
        self.save_dir_eval = save_dir_eval
        self.algo_type = algo_type
        
        # SE parameters
        self.se_params = {'model_path':None,
                      'device':device,
                      'nmf_rank':nmf_rank,
                      'num_iter':num_iter,
                      'num_E_step':num_E_step,
                      'use_visual_feature_extractor': False}

        if not os.path.isdir(self.save_dir_eval):
            os.makedirs(self.save_dir_eval)
            self.build_eval_data()
    
    def build_eval_data(self):
        
        fs = 16000
        meter = pyln.Meter(fs) # create BS.1770 meter

        speech_file_list = []
        for speaker_id in self.speakers_id:
            speech_file_list.append(os.path.join(self.speech_dir, speaker_id, 'sa1.wav'))
            
        cnt = 0
        
        for speech_file in speech_file_list:

            this_clean_wav, fs = sf.read(speech_file) 
            utt_len = len(this_clean_wav)
            path0, file_name = os.path.split(speech_file)
            path1, speaker_id = os.path.split(path0)

            video_file = os.path.join(self.speech_dir, speaker_id, file_name[:-4]+'RawVF.npy')
            video_raw = np.load(video_file)  # video
            
            for noise_type in self.noise_types:

                for target_snr in self.SNRs:

                    noise_file = os.path.join(self.demand_noise_dir, noise_type, 'ch01.wav')
                    this_noise_wav, fs = sf.read(noise_file)
                    noise_start = np.random.randint(0, len(this_noise_wav) - utt_len - 1)
                    this_noise_wav = this_noise_wav[noise_start:noise_start+utt_len]

                    s_loudness = meter.integrated_loudness(this_clean_wav)
                    n_loudness = meter.integrated_loudness(this_noise_wav)

                    input_snr = s_loudness - n_loudness
                    scale_factor = 10**( (input_snr - target_snr)/20 )
                    this_noise_wav = this_noise_wav*scale_factor
                    
                    eval_file_dic = {}
                    eval_file_dic = {'speech_wav': this_clean_wav,
                                     'noise_wav': this_noise_wav,
                                     'noise_type': noise_type,
                                     'video_raw': video_raw,
                                     'snr': target_snr}
                    
                    save_dict(eval_file_dic, os.path.join(self.save_dir_eval, 'eval_idx_' + str(cnt) + '_noise_' + noise_type + '_snr_' + str(target_snr) + '.pkl'))
                    
                    cnt += 1   
                    
    def setup_se(self, model_path):
        
        self.se_params['model_path'] = model_path
        self.se = SpeechEnhancement(self.se_params)
        
    def run_se(self, input_data_path):
        
        input_data = load_dict(os.path.join(self.save_dir_eval, input_data_path))
        speech_wav = input_data['speech_wav']
        noise_wav = input_data['noise_wav']
        video_raw = input_data['video_raw']
        mix_wav = speech_wav + noise_wav
        
        # Run SE & evaluations
        info = self.se.run([mix_wav, speech_wav, video_raw, self.algo_type])
        
        return info
        
class LearningAlgorithm():

    """
    Basical class for model building, including:
    - read common paramters for different models
    - define data loader
    - define loss function as a class member
    """

    def __init__(self, config_file='config_default.ini', use_visual_feature_extractor = True):
        
        # Load config parser
        self.config_file = config_file
        if not os.path.isfile(self.config_file):
            raise ValueError('Invalid config file path')

        self.cfg = myconf()
        
        self.cfg.read(self.config_file)

        self.hparams = ini_to_dict(self.config_file)
        
        self.default_params = {'rnn' : 'LSTM', 'inference' : 'gated', 'overlap' : 1, "v_vae_path" : None, "pretrained_model" : None, "shuffle_file_list" : True, "shuffle_samples_in_batch" : True}

        self.model_name = self.cfg.get('Network', 'name')
        self.vae_mode = self.cfg.get('Network', 'vae_mode')
        self.rnn_cell = self.cfg.get('Network', 'rnn', fallback = self.default_params['rnn'])
        self.inference_type = self.cfg.get('Network', 'inference', fallback = self.default_params['inference'])
        self.v_vae_path = self.cfg.get('Network', 'v_vae_path', fallback = self.default_params['v_vae_path'])
        self.pretrained_model = self.cfg.get('Network', 'pretrained_model', fallback = self.default_params['pretrained_model'])


        self.dataset_name = self.cfg.get('DataFrame', 'dataset_name')
        self.overlap = self.cfg.getfloat('DataFrame', 'overlap', fallback = self.default_params['overlap'])
        self.demand_noise_dir = self.cfg.get('User', 'demand_noise_dir')

        self.use_visual_feature_extractor = use_visual_feature_extractor

        # Get host name and date
        self.hostname = socket.gethostname()
        self.date = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%M")
        self.api_key = self.cfg.get('User', 'api_key')
        
        # Load STFT parameters
        wlen_sec = self.cfg.getfloat('STFT', 'wlen_sec')
        hop_percent = self.cfg.getfloat('STFT', 'hop_percent')
        fs = self.cfg.getint('STFT', 'fs')
        zp_percent = self.cfg.getint('STFT', 'zp_percent')
        wlen = wlen_sec * fs
        wlen = np.int(np.power(2, np.ceil(np.log2(wlen)))) # pwoer of 2
        hop = np.int(hop_percent * wlen)
        nfft = wlen + zp_percent * wlen
        win = np.sin(np.arange(0.5, wlen+0.5) / wlen * np.pi)

        STFT_dict = {}
        STFT_dict['fs'] = fs
        STFT_dict['wlen'] = wlen
        STFT_dict['hop'] = hop
        STFT_dict['nfft'] = nfft
        STFT_dict['win'] = win
        STFT_dict['trim'] = self.cfg.getboolean('STFT', 'trim')
        self.STFT_dict = STFT_dict

        self.audioSpec_logger = cometml_logger.AudioSpecLogger(STFT_dict = self.STFT_dict)
            
        # Load model parameters
        self.use_cuda = self.cfg.getboolean('Training', 'use_cuda')
        self.device = 'cuda' if torch.cuda.is_available() and self.use_cuda else 'cpu'
        
        # Build model
        self.build_model()


    def build_model(self):

        if self.model_name == 'VAE':
            self.model = build_VAE(cfg=self.cfg, device=self.device, vae_mode = self.vae_mode, exp_mode = 'train', v_vae_path = self.v_vae_path)
        elif self.model_name == 'DKF':
            self.model = build_DKF(cfg=self.cfg, device=self.device, vae_mode = self.vae_mode)

        if self.use_visual_feature_extractor:
            self.vfeats = self.build_visual_extractor()

    def init_optimizer(self):

        # Load
        self.optimization  = self.cfg.get('Training', 'optimization')
        lr = self.cfg.getfloat('Training', 'lr')

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
    def build_dataloader(self, train_data_dir, val_data_dir, sequence_len, batch_size, STFT_dict, use_random_seq=False, overlap=1, vf_root = None, extract_and_save_visual_feats = False):

        # List all the data with certain suffix
        data_suffix = self.cfg.get('DataFrame', 'suffix')
        train_file_list = librosa.util.find_files(train_data_dir, ext=data_suffix)
        val_file_list = librosa.util.find_files(val_data_dir, ext=data_suffix)
        # Generate dataloader for pytorch
        num_workers = self.cfg.getint('DataFrame', 'num_workers')
        shuffle_file_list = self.cfg.get('DataFrame', 'shuffle_file_list', fallback = self.default_params['shuffle_file_list'])
        shuffle_samples_in_batch = self.cfg.get('DataFrame', 'shuffle_samples_in_batch', fallback = self.default_params['shuffle_samples_in_batch'])
        data_dir = self.cfg.get('User', 'data_dir')
        
        train_file_list = [os.path.join(root, name) for root, dirs, files in os.walk(os.path.join(data_dir, 'train_data_NTCD')) for name in files if name.endswith('.wav')]

        val_file_list = [os.path.join(root, name) for root, dirs, files in os.walk(os.path.join(data_dir, 'val_data_NTCD')) for name in files if name.endswith('.wav')]
        
        if self.model_name == 'VAE':
            train_dataset = SpeechDatasetFrames(file_list=train_file_list, sequence_len=sequence_len,
                                                  STFT_dict=self.STFT_dict, shuffle=shuffle_file_list, name=self.dataset_name, vf_root = vf_root)
            val_dataset = SpeechDatasetFrames(file_list=val_file_list, sequence_len=sequence_len,
                                                    STFT_dict=self.STFT_dict, shuffle=shuffle_file_list, name=self.dataset_name, vf_root = vf_root)

        elif self.model_name == 'DKF':
            train_dataset = SpeechSequencesFull(file_list=train_file_list, sequence_len=sequence_len, overlap=overlap,
                                                  STFT_dict=self.STFT_dict, shuffle=shuffle_file_list, name=self.dataset_name, gaussian = 0, vf_root = vf_root, extract_visual_features = False)
            val_dataset = SpeechSequencesFull(file_list=val_file_list, sequence_len=sequence_len, overlap=overlap,
                                                    STFT_dict=self.STFT_dict, shuffle=True, name=self.dataset_name, gaussian = 0, vf_root = vf_root, extract_visual_features = False)


        # Extract and save visual features a single time:
        if extract_and_save_visual_feats:
            print('Extracting and saving visual features ... ')
            test_file_list = [os.path.join(root, name) for root, dirs, files in os.walk(os.path.join(data_dir, 'test_data_NTCD', 'clean')) for name in files if name.endswith('.wav')]
            SpeechDatasetSequences(file_list=test_file_list, sequence_len=sequence_len, overlap=1,
                                                    STFT_dict=self.STFT_dict, shuffle=True, name=self.dataset_name, extract_visual_features = True)
            SpeechDatasetSequences(file_list=train_file_list, sequence_len=sequence_len, overlap=1,
                                                    STFT_dict=self.STFT_dict, shuffle=True, name=self.dataset_name, extract_visual_features = True)     
            SpeechDatasetSequences(file_list=val_file_list, sequence_len=sequence_len, overlap=1,
                                                    STFT_dict=self.STFT_dict, shuffle=True, name=self.dataset_name, extract_visual_features = True)
            
        train_num = train_dataset.__len__()
        val_num = val_dataset.__len__()


        # Create dataloader
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                       shuffle=shuffle_samples_in_batch,
                                                       num_workers = num_workers)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                     shuffle=True,
                                                     num_workers = num_workers)

        return train_dataloader, val_dataloader, train_num, val_num


    def get_basic_info(self):

        basic_info = []
        basic_info.append('HOSTNAME: ' + self.hostname)
        basic_info.append('Time: ' + self.date)
        basic_info.append('Device for training: ' + self.device)
        if self.device == 'cuda':
            basic_info.append('Cuda verion: {}'.format(torch.version.cuda))
        basic_info.append('Model name: {}'.format(self.model_name))
        basic_info.append('VAE mode: {}'.format(self.vae_mode))

        return basic_info


    def train(self, vf_root = None):

        # Set module.training = True
        self.model.train()
        torch.autograd.set_detect_anomaly(True)

        # Create directory for results
        saved_root = self.cfg.get('User', 'saved_root')
        z_dim = self.cfg.getint('Network','z_dim')
        tag = self.cfg.get('Network', 'tag')
        filename = "{}_{}_{}_z_dim={}".format(self.dataset_name, self.date, tag, z_dim)
        save_dir = os.path.join(saved_root, filename)
        if not(os.path.isdir(save_dir)):
            os.makedirs(save_dir)
        print("Save Dir: ", save_dir)
        # Save the model configuration
        save_cfg = os.path.join(save_dir, 'config.ini')
        shutil.copy(self.config_file, save_cfg)

        # Create logger
        log_file = os.path.join(save_dir, 'log.txt')
        logger_type = self.cfg.getint('User', 'logger_type')
        logger = get_logger(log_file, logger_type)

        # Print basic infomation
        for log in self.get_basic_info():
            logger.info(log)
        logger.info('In this experiment, the results will be saved in: ' + save_dir)

        # Print model infomation (optional)
        if self.cfg.getboolean('User', 'print_model'):
            for log in self.model.get_info():
                logger.info(log)

        # Init optimizer
        self.init_optimizer()

        batch_size = self.cfg.getint('Training', 'batch_size')
        sequence_len = self.cfg.getint('DataFrame','sequence_len')
        use_random_seq = self.cfg.getboolean('DataFrame','use_random_seq')

        # Create data loader
        train_data_dir = self.cfg.get('User', 'train_data_dir')
        val_data_dir = self.cfg.get('User', 'val_data_dir')
        loader = self.build_dataloader(train_data_dir=train_data_dir, val_data_dir=val_data_dir,
                                       sequence_len=sequence_len, batch_size=batch_size,
                                       STFT_dict=self.STFT_dict, use_random_seq=use_random_seq, overlap = self.overlap, vf_root = vf_root)
        train_dataloader, val_dataloader, train_num, val_num = loader
        log_message = 'Training samples: {}'.format(train_num)
        logger.info(log_message)
        print(log_message)
        log_message = 'Validation samples: {}'.format(val_num)
        logger.info(log_message)
        print(log_message)

        path = self.train_normal(logger, save_dir, train_dataloader, val_dataloader, train_num, val_num)
        
        return path
        
    def train_normal(self, logger, save_dir, train_dataloader, val_dataloader, train_num, val_num):

        # Setup val data for SE validation
        fs = self.cfg.getint('STFT', 'fs')
        data_dir = self.cfg.get('User', 'data_dir')
        demand_noise_dir = self.cfg.get('User', 'demand_noise_dir')
        save_dir_eval = os.path.join(data_dir, 'val_se')
        val_se_file_list = [f for f in os.listdir(save_dir_eval) if f.endswith('.pkl')]
        self.se_eval = SE_Eval(data_dir = data_dir,
                               demand_noise_dir = demand_noise_dir,
                               save_dir_eval = save_dir_eval)
        
        # Load training parameters
        epochs = self.cfg.getint('Training', 'epochs')
        early_stop_patience = self.cfg.getint('Training', 'early_stop_patience')
        save_frequency = self.cfg.getint('Training', 'save_frequency')
        log_output = self.cfg.get('User', 'log_output')
        
        #reload model
        if self.pretrained_model is not None:
            checkpoint = torch.load(self.pretrained_model)
            if 'model_state_dict' in checkpoint.keys():
                checkpoint = checkpoint['model_state_dict']
            print("Loading Model ...")
            self.model.load_state_dict(checkpoint, strict=False)
        
        # Setup Comet ML log
        iter_meter = IterMeter()
        experiment = Experiment(self.api_key, project_name=self.hparams['User']['project_name'])
        experiment.log_parameters(self.hparams)
        experiment_name = '_'.join(self.hparams['User']['cometml_tags'].split(","))
        
        for tag in self.hparams['User']['cometml_tags'].split(","):
            experiment.add_tag(tag)
        if self.hparams['User']['experiment_name'] is not None:
            experiment.set_name(self.hparams['User']['experiment_name'])
        else:
            experiment.set_name(experiment_name)
    
        # Create python list for loss
        train_loss = np.zeros((epochs,))
        val_loss = np.zeros((epochs,))
        train_recon = np.zeros((epochs,))
        train_KLD = np.zeros((epochs,))
        val_recon = np.zeros((epochs,))
        val_KLD = np.zeros((epochs,))
        best_val_loss = np.inf
        cpt_patience = 0
        cur_best_epoch = epochs
        best_state_dict = self.model.state_dict()

        # Define optimizer (might use different training schedule)
        optimizer = self.optimizer

        # SE metrics ("i": improvement with respect to input)
        metrics_names = ['si_sdr_i', 'pesq_i', 'stoi_i'] 
        
        # Train with mini-batch SGD
        for epoch in range(epochs):

            res_dic = {}
            for metric_name in metrics_names:
                res_dic[metric_name] = {'mean': 0., 'median': 0., 'std': 0., 'acc': []}
                
            start_time = datetime.datetime.now()
    
            # Batch training
            for batch_idx, (batch_a, batch_v, _) in tqdm(enumerate(train_dataloader)):

                batch_a, batch_v = batch_a.to(self.device), batch_v.to(self.device)

                self.model(batch_a, batch_v, compute_loss=True)

                loss_tot, loss_recon, loss_KLD = self.model.loss

                optimizer.zero_grad()
                loss_tot.backward()
                optimizer.step()

                train_loss[epoch] += loss_tot.item()
                train_recon[epoch] += loss_recon.item()
                train_KLD[epoch] += loss_KLD.item()
        
            # Validation - reconstruction loss
            for batch_idx, (batch_a, batch_v, batch_phase) in tqdm(enumerate(val_dataloader)):

                batch_a, batch_v = batch_a.to(self.device), batch_v.to(self.device)

                batch_a_recon = self.model(batch_a, batch_v, compute_loss=True)

                loss_tot, loss_recon, loss_KLD = self.model.loss

                val_loss[epoch] += loss_tot.item()
                val_recon[epoch] += loss_recon.item()
                val_KLD[epoch] += loss_KLD.item()

                if log_output:
                    batch_phase = batch_phase.to(self.device)
                    self.audioSpec_logger.log_audio_spectrogram(
                        batch_a.detach(),
                        batch_a_recon.detach(),
                        batch_phase.detach(),
                        experiment, step=iter_meter.get(), tag='validation_data_', max_batch_items=5)
                
            # Loss normalization
            train_loss[epoch] = train_loss[epoch]/ train_num
            val_loss[epoch] = val_loss[epoch] / val_num
            train_recon[epoch] = train_recon[epoch] / train_num
            train_KLD[epoch] = train_KLD[epoch]/ train_num
            val_recon[epoch] = val_recon[epoch] / val_num
            val_KLD[epoch] = val_KLD[epoch] / val_num

            # Early stop patiance
            if early_stop_patience > 0:
            
                if val_loss[epoch] < best_val_loss:
                    best_val_loss = val_loss[epoch]
                    cpt_patience = 0
                    best_state_dict = self.model.state_dict()
                    cur_best_epoch = epoch
                else:
                    cpt_patience += 1

                # Stop traning if early-stop triggers
                if cpt_patience == early_stop_patience:
                    logger.info('Early stop patience achieved')
                    print('Early stopping occured ...')
                    break


            # Save model parameters regularly
            if epoch % save_frequency == 0:
                save_file = os.path.join(save_dir, self.model_name + '-' + self.vae_mode + '_epoch' + str(epoch) + '.pt')
                torch.save(self.model.state_dict(), save_file)
                
                # Validation - speech enhancement metrics
                self.se_eval.setup_se(save_file)                                
                for idx_eval, this_wav_path in tqdm(enumerate(val_se_file_list)):
                    info_out = self.se_eval.run_se(this_wav_path)
                    # Compute scores
                    metrics_dict = get_metrics(mix = info_out['noisy_wave'], clean = info_out['clean_wave'], estimate = info_out['enh_wave'],
                                               sample_rate=fs, metrics_list=['si_sdr', 'stoi', 'pesq'])

                    input_scores = [metrics_dict['input_si_sdr'], metrics_dict['input_pesq'], metrics_dict['input_stoi']]
                    output_scores = [metrics_dict['si_sdr'], metrics_dict['pesq'], metrics_dict['stoi']]

                    res_dic['si_sdr_i']['acc'].append(output_scores[0] - input_scores[0])
                    res_dic['pesq_i']['acc'].append(output_scores[1] - input_scores[1])
                    res_dic['stoi_i']['acc'].append(output_scores[2] - input_scores[2])

                res_dic = cometml_logger.report_se_metrics(res_dic, experiment, iter_meter.get(), iter_meter.get())  
            
            experiment.log_metric('train/recon', train_recon[epoch], step=iter_meter.get())
            experiment.log_metric('train/KLD', train_KLD[epoch], step=iter_meter.get())
            experiment.log_metric('train/total', train_loss[epoch], step=iter_meter.get())

            experiment.log_metric('val/recon', val_recon[epoch], step=iter_meter.get())
            experiment.log_metric('val/KLD', val_KLD[epoch], step=iter_meter.get())
            experiment.log_metric('val/total', val_loss[epoch], step=iter_meter.get())
            
            iter_meter.step()
            
            # Training time
            end_time = datetime.datetime.now()
            interval = (end_time - start_time).seconds / 60
            log_message = 'Epoch: {} train loss: {:.4f} val loss {:.4f} training time {:.2f}m'.format(epoch, train_loss[epoch], val_loss[epoch], interval)
            logger.info(log_message)
            print(log_message)

        
        # Save the final weights of network with the best validation loss
        train_loss = train_loss[:epoch+1]
        val_loss = val_loss[:epoch+1]
        train_recon = train_recon[:epoch+1]
        train_KLD = train_KLD[:epoch+1]
        val_recon = val_recon[:epoch+1]
        val_KLD = val_KLD[:epoch+1]
        save_file = os.path.join(save_dir, self.model_name + '-' + self.vae_mode + '_final_epoch' + str(epoch) + '.pt')
        torch.save(best_state_dict, save_file)

        # Save the training loss and validation loss
        loss_file = os.path.join(save_dir, 'loss_model.pckl')
        with open(loss_file, 'wb') as f:
            pickle.dump([train_loss, val_loss, train_recon, train_KLD, val_recon, val_KLD], f)


        # Save the loss figure
        tag = self.vae_mode
        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(train_loss, label='training loss')
        plt.plot(val_loss, label='validation loss')
        plt.legend(fontsize=16, title=self.model_name, title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        fig_file = os.path.join(save_dir, 'loss_{}.png'.format(tag))
        plt.savefig(fig_file)

        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(train_recon, label='Reconstruction')
        plt.plot(train_KLD, label='KL Divergence')
        plt.legend(fontsize=16, title='{}: Training'.format(self.model_name), title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        fig_file = os.path.join(save_dir, 'loss_train_{}.png'.format(tag))
        plt.savefig(fig_file)

        plt.clf()
        fig = plt.figure(figsize=(8,6))
        plt.rcParams['font.size'] = 12
        plt.plot(val_recon, label='Reconstruction')
        plt.plot(val_KLD, label='KL Divergence')
        plt.legend(fontsize=16, title='{}: Validation'.format(self.model_name), title_fontsize=20)
        plt.xlabel('epochs', fontdict={'size':16})
        plt.ylabel('loss', fontdict={'size':16})
        fig_file = os.path.join(save_dir, 'loss_val_{}.png'.format(tag))
        plt.savefig(fig_file)

        return save_file


    def generate_prior(self, sample_length, audio_file = None, state_dict_file=None, save_flag = False, seed = 0, save_video = False):

        mean = torch.zeros(sample_length, self.model.z_dim).to(self.device)
        logvar = torch.zeros(sample_length, self.model.z_dim).to(self.device)
        z_prior = self.model.reparameterization(mean, logvar)
        y_prior = self.model.generation_x(z_prior)
        print(y_prior.shape)
        return y_prior.t().detach().cpu().numpy()


    def generate(self, audio_orig, audio_recon=None, state_dict_file=None, save_flag = False, denoise = False, target_snr = 10, seed = 0, model_type = 'A-VAE', save_video = False, trim_s = True, generate_prior = True):
        """
        Input: a reference audio (and a predefined path for generated audio
        Output: generated audio
        """
        # Define generated
        if audio_recon == None:
            #print('Generated audio file will be saved in the same directory as reference audio')
            audio_dir, audio_file = os.path.split(audio_orig)
            file_name, file_ext = os.path.splitext(audio_file)
            audio_recon = os.path.join(audio_dir, file_name+'_recon'+file_ext)

        root_dir, filename = os.path.split(audio_recon)
        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)

        # Load model state
        if state_dict_file != None:
            self.model.load_state_dict(torch.load(state_dict_file, map_location=self.device))
            # Set module.training = False
            self.model.eval()


        # Read STFT parameters
        fs = self.STFT_dict['fs']
        nfft = self.STFT_dict['nfft']
        hop = self.STFT_dict['hop']
        wlen = self.STFT_dict['wlen']
        win = np.sin(np.arange(0.5, wlen+0.5) / wlen * np.pi)
        trim = self.STFT_dict['trim']

        # Read original audio file
        x, fs_x = sf.read(audio_orig)
        if fs != fs_x:
            raise ValueError('Unexpected sampling rate')

        path, fname = os.path.split(audio_orig)
        videofile = os.path.join(path, fname[:-4]+'Raw.npy')
        v_orig = np.load(videofile)
        data_v_to_save = v_orig.copy().transpose().reshape((-1,1,67,67,1))
        X, v_up, T_orig = extract_frames(x, v_orig, self.STFT_dict, trim_s, extract_visual_features = True, fs = fs, vfeats = self.vfeats)
        # Prepare data input
        data_orig = np.abs(X) ** 2 # (x_dim, seq_len)


        if trim_s:
            x, _ = librosa.effects.trim(x, top_db=30)

        fps = 30


        if save_video:
            out_path = os.path.join(root_dir,os.path.splitext(os.path.split(audio_recon)[-1])[0]+'_video.avi')
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'DIVX'), fps,(67,67))
            for idx in tqdm(range(data_v_to_save.shape[0])):
                frame = cv2.cvtColor(np.uint8(data_v_to_save[idx, 0, :, :, 0].transpose()*255),cv2.COLOR_GRAY2BGR)
                out.write(frame)
            out.release()
            cv2.destroyAllWindows()


        data_orig = torch.from_numpy(data_orig.astype(np.float32)).to(self.device)
        v_orig = torch.from_numpy(v_up.astype(np.float32)).to(self.device)


        # Reconstruction
        with torch.no_grad():
            if model_type == 'A-VAE':
                data_recon = self.model(data_orig.t(), compute_loss=False, exp_mode="test").to('cpu').detach().numpy().T
                _,z_mean, _ = self.model.inference(data_orig.t()) # corresponds to the encoder mean
                mean = torch.zeros(data_orig.t().shape[0], self.model.z_dim).to(self.device)
                logvar = torch.zeros(data_orig.t().shape[0], self.model.z_dim).to(self.device)
                z_prior = self.model.reparameterization(mean, logvar)

            elif model_type == "V-VAE":
                data_recon = self.model(data_orig.t(), v_orig, compute_loss=False).to('cpu').detach().numpy().T
                _,z_mean, _ = self.model.inference(data_orig.t(), v_orig) # corresponds to the encoder mean
            elif model_type == "AV-VAE":
                data_recon = self.model(data_orig.t(), v_orig, compute_loss=False).to('cpu').detach().numpy().T
                _,z_mean, _ = self.model.inference(data_orig.t(), v_orig) # corresponds to the encoder mean
            elif model_type == "AV-CVAE":
                print(data_orig.shape, v_orig.shape)
                data_recon = self.model(data_orig.t(), v_orig, compute_loss=False).to('cpu').detach().numpy().T
                _,z_mean, _ = self.model.inference(data_orig.t(), v_orig) # corresponds to the encoder mean
                _, _, z_prior = self.model.zprior(v_orig)
                prior_recon = self.model.generation_x(z_prior, v_orig).detach().cpu().numpy().T

            z_mean = z_mean.to('cpu').detach().numpy()


        # Re-synthesis
        X_recon_nophase = np.sqrt(data_recon.copy())# * np.exp(1j * np.angle(X))
        x_recon_nophase = librosa.istft(X_recon_nophase, hop_length=hop, win_length=wlen, window=win)
        if generate_prior:
            P_recon_nophase = np.sqrt(prior_recon.copy())# * np.exp(1j * np.angle(X))
            p_recon_nophase = librosa.istft(P_recon_nophase, hop_length=hop, win_length=wlen, window=win)
            P_recon = np.sqrt(prior_recon) * np.exp(1j * np.angle(X))
            p_recon = librosa.istft(P_recon, hop_length=hop, win_length=wlen, window=win)

        X_recon = np.sqrt(data_recon) * np.exp(1j * np.angle(X))
        x_recon = librosa.istft(X_recon, hop_length=hop, win_length=wlen, window=win)


        # Wrtie audio file
        scale_norm_nophase = 1 / np.max(np.abs(x_recon_nophase)) * 0.9
        if generate_prior:
            scale_prior_nophase = 1 / np.max(np.abs(p_recon_nophase)) * 0.9
            scale_prior = 1 / np.max(np.abs(p_recon)) * 0.9
        scale_norm = 1 / np.max(np.abs(x_recon)) * 0.9
        if save_flag == True:
            sf.write(audio_recon, scale_norm*x_recon, fs_x)
            path, name = os.path.split(audio_recon)
            sf.write(os.path.join(path, name[:-4]+'-nophase.wav'), scale_norm_nophase*x_recon_nophase, fs_x)
            sf.write(os.path.join(path, name[:-4]+'-original.wav'), x, fs_x)
            if save_video:
                os.system(f"ffmpeg -i {out_path} -i {audio_recon} -c:v copy -c:a aac {os.path.join(path, name[:-4]+'-video.avi')} -y")
                os.system(f"ffmpeg -i {out_path} -i {os.path.join(path, name[:-4]+'-nophase.wav')} -c:v copy -c:a aac {os.path.join(path, name[:-4]+'-video-nophase.avi')} -y")
                os.system(f"ffmpeg -i {out_path} -i {os.path.join(path, name[:-4]+'-original.wav')} -c:v copy -c:a aac {os.path.join(path, name[:-4]+'-video-original.avi')} -y")
            if generate_prior:
                sf.write(os.path.join(path, name[:-4]+'-prior_nophase.wav'), scale_prior_nophase*p_recon_nophase, fs_x)
                sf.write(os.path.join(path, name[:-4]+'-prior.wav'), scale_prior*p_recon, fs_x)

        metrics_dict = get_metrics(mix = x, clean = x, est = scale_norm*x_recon, sample_rate=fs, metrics_list=['si_sdr', 'stoi', 'pesq'])
        
        return metrics_dict

    def build_visual_extractor(self):
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

        vfeats = Lipreading( modality='video',
                            num_classes=500,
                            tcn_options=tcn_options,
                            backbone_type=backbone_type,
                            relu_type=relu_type,
                            width_mult=width_mult,
                            extract_feats=True).cuda()
        vfeats = load_model("./lipreading/data/lrw_resnet18_mstcn_adamw_s3.pth.tar", vfeats, allow_size_mismatch=False)
        vfeats.eval()

        return vfeats

    def generate_dkf(self, audio_orig, audio_recon=None, state_dict_file=None, save_flag = False):
        """
        Input: a reference audio (and a predefined path for generated audio
        Output: generated audio
        """

        # Define generated
        if audio_recon == None:
            #print('Generated audio file will be saved in the same directory as reference audio')
            audio_dir, audio_file = os.path.split(audio_orig)
            file_name, file_ext = os.path.splitext(audio_file)
            audio_recon = os.path.join(audio_dir, file_name+'_recon'+file_ext)
        else:
            root_dir, filename = os.path.split(audio_recon)
            if not os.path.isdir(root_dir):
                os.makedirs(root_dir)

        # Load model state
        if state_dict_file != None:
            self.model.load_state_dict(torch.load(state_dict_file, map_location=self.device))

        # Read STFT parameters
        fs = self.STFT_dict['fs']
        nfft = self.STFT_dict['nfft']
        hop = self.STFT_dict['hop']
        wlen = self.STFT_dict['wlen']
        win = np.sin(np.arange(0.5, wlen+0.5) / wlen * np.pi)
        trim = self.STFT_dict['trim']

        # Read original audio file
        x, fs_x = sf.read(audio_orig)
        if fs != fs_x:
            raise ValueError('Unexpected sampling rate')

        # Silence triming
        #if trim:
        #    x, _ = librosa.effects.trim(x, top_db=30)

        # Scaling
        scale = np.max(np.abs(x))
        x = x / scale

        # STFT
        X = librosa.stft(x, n_fft=nfft, hop_length=hop, win_length=wlen, window=win)

        # Read video file
        path, fname = os.path.split(audio_orig)
        videofile = os.path.join(path, fname[:-4]+'Raw.npy')
        v_orig = np.load(videofile)

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

            t1 = 1
            t2 = -1
            T_orig = len(x)/fs
            fps = 30

            v1, v2 = int(np.floor(t1*fps)), int(np.floor((T_orig-t2)*fps))

            v_trimmed = v_orig[:,v1:N_vframes-v2]

            data_v = resample(v_trimmed, target_num = N_aframes)
            data_v = data_v.transpose().reshape((-1,1,67,67,1))
            data_v = torch.from_numpy(data_v).permute(1,-1,0,2,3).to('cuda').type(torch.cuda.FloatTensor)
            data_orig_v = self.vfeats(data_v, lengths=None)[0,...].detach().cpu().numpy()
            data_orig_v = torch.from_numpy(data_orig_v.astype(np.float32).transpose()).to(self.device)

        data_orig = np.abs(X) ** 2 # (x_dim, seq_len)
        data_orig_a = torch.from_numpy(data_orig.astype(np.float32)).to(self.device)

        self.model.eval()

        # Reconstruction
        with torch.no_grad():
            data_recon = self.model(data_orig_a, data_orig_v, compute_loss=False).to('cpu').detach().numpy()


        # Re-synthesis
        X_recon = np.sqrt(data_recon) * np.exp(1j * np.angle(X))
        x_recon = librosa.istft(X_recon, hop_length=hop, win_length=wlen, window=win)

        # Wrtie audio file
        scale_norm = 1 / (np.maximum(np.max(np.abs(x_recon)), np.max(np.abs(x)))) * 0.9

        if save_flag == True:
            sf.write(audio_recon, scale_norm*x_recon, fs_x)

        metrics_dict = get_metrics(mix = x, clean = x, est = scale_norm*x_recon, sample_rate=fs, metrics_list=['si_sdr', 'stoi', 'pesq'])
        
        return metrics_dict


    def test(self, data_dir, state_dict_file=None, recon_dir=None):
        """
        Apply re-synthesis for all audio files in a given directory, and return evaluation results
        All generated audio files in the same root as data_dir, named audio_dir + '_{}_recon'.format(tag)
        One could use state_dict_file to load preserved model state, otherwise it will use model state in cache
        Attention: if there already exist a folder with the same name, it will be deleted
        """

        # Remove '/' in the end if exist
        if data_dir[-1] == '/':
            data_dir = data_dir[:-1]
        else:
            data_dir = data_dir

        # Find all audio files
        data_suffix = self.cfg.get('DataFrame', 'suffix')
        audio_list = librosa.util.find_files(data_dir, ext=data_suffix)

        # Create re-synthesis folder
        if recon_dir == None:
            tag = self.cfg.get('Network', 'tag')
            root, audio_dir = os.path.split(data_dir)
            recon_dir = os.path.join(root, audio_dir + '_{}_recon-utt'.format(tag))
            if os.path.isdir(recon_dir):
                shutil.rmtree(recon_dir)
            os.mkdir(recon_dir)
        else:
            print('Saved dir pre-defined')
        print('Re-synthesis results will be saved in {}'.format(recon_dir))

        # Load model state, avoid repeatly loading model state
        if state_dict_file == None:
            print('Use model state in cache...')
        else:
            print('Model state loading...')
