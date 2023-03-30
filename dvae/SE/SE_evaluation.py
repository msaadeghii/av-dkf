#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Speech enhancement evaluation on the test set of NTCD-TIMIT dataset.

@author: Mostafa Sadeghi (mostafa.sadeghi@inria.fr)
"""

from argparse import ArgumentParser
from speech_enhancement import SpeechEnhancement
import os
import torch
import sys
import time
import numpy as np
from tqdm import tqdm
import json

from six.moves import cPickle as pickle #for performance

from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass


def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

        
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def init_params(params):
    
    # SE parameters
    se_params = {'device':device,
              'nmf_rank':params.nmf_rank,
              'num_iter':params.num_iter,
              'num_E_step':params.num_E_step,
              'model_path':params.ckpt_path,
              'save_dir':params.save_dir,
              'save_flg':True,
              'use_visual_feature_extractor': False}

    se = SpeechEnhancement(se_params)
    
    return se

if __name__ == '__main__':
    
    parser = ArgumentParser(description='Parallel enhancement of speech signals.')
    parser.add_argument("--algo_type", type=str, required=True, help='Type of enhancement algorithm, e.g., PEEM, GPEEM')
    parser.add_argument("--ckpt_path", type=str, required=True, help='Path to the checkpoint model.')
    parser.add_argument("--segment", type=int, default=-1, help='Segment of the test files to evaluate on.')
    parser.add_argument("--nmf_rank", type=int, default=8, help='NMF rank for noise model.')
    parser.add_argument("--num_iter", type=int, default=100, help='Number of EM iterations for the enhancement algorithm.')
    parser.add_argument("--num_E_step", type=int, default=10, help='Number of iterations for the E step.')
    parser.add_argument("--data_dir", type=str, required=True, help='Directory to the test data.')
    parser.add_argument("--save_root", type=str, required=True, help='Directory to save the results.')
    args = parser.parse_args()

    save_dir = os.path.join(args.save_root, os.path.basename(args.ckpt_path)[:-3], args.algo_type)
    os.makedirs(save_dir, exist_ok = True)
    args.save_dir = save_dir
    
    # save the input args
    args_file = f'{args.save_dir}/commandline_args.txt'
    with open(args_file, 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    # start the timer
    start_time = time.time()

    # load file list and select the target segment to process
    files_list = load_dict(args.data_dir)
    seg_id = args.segment
    
    if seg_id >= 0:
        num_files = len(files_list)
        files_list = files_list[seg_id * (num_files // 3):(seg_id+1) * (num_files // 3)]
    
    print(f'Total number of files to process: {len(files_list)}') 
    
    # initialize the speech enhancement model
    se = init_params(args)
  
    # parallel processing 
    input_params = [[filename, args.algo_type] for filename in files_list]
    chunksize = 10
    with Pool(4) as pool:
        result = pool.map(se.run, input_params, chunksize)
        
    # stop the timer
    end_time = time.time()

    # calculate the elapsed time
    elapsed_time = end_time - start_time

    # close the pool of processes
    pool.close()
    pool.join()
    
    print(f'Elapsed time for a total of {len(files_list)} files is {elapsed_time}') 