'''
Adapted from original code by Clarity Challenge
https://github.com/claritychallenge/clarity
'''

import os
from tqdm import tqdm
import pandas as pd
from soundfile import SoundFile
from concurrent.futures import ProcessPoolExecutor
from argparse import ArgumentParser
import numpy as np
from six.moves import cPickle as pickle #for performance
from pesq import pesq
from pystoi import stoi

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def compute_sisdr(reference, estimate):
    """ Compute the scale invariant SDR.

    Parameters
    ----------
    estimate : array of float, shape (n_samples,)
        Estimated signal.
    reference : array of float, shape (n_samples,)
        Ground-truth reference signal.

    Returns
    -------
    sisdr : float
        SI-SDR.
        
    Example
    --------
    >>> import numpy as np
    >>> from sisdr_metric import compute_sisdr
    >>> np.random.seed(0)
    >>> reference = np.random.randn(16000)
    >>> estimate = np.random.randn(16000)
    >>> compute_sisdr(estimate, reference)
    -48.1027283264049    
    """
    eps = np.finfo(estimate.dtype).eps
    alpha = (np.sum(estimate*reference) + eps) / (np.sum(np.abs(reference)**2) + eps)
    sisdr = 10*np.log10((np.sum(np.abs(alpha*reference)**2) + eps)/
                        (np.sum(np.abs(alpha*reference - estimate)**2) + eps))
    return sisdr

def compute_pesq(target, enhanced, sr, mode = 'wb'):
    """Compute PESQ from: https://github.com/ludlows/python-pesq/blob/master/README.md
        Args:
            target (string): Name of file to read
            enhanced (string): Name of file to read
            sr (int): sample rate of files
            mode (string): 'wb' = wide-band (16KHz); 'nb' narrow-band (8KHz)
        Returns:
            PESQ metric (float)
                """
    return pesq(sr, target, enhanced, mode)

def compute_stoi(target, enhanced, sr):
    """Compute STOI from: https://github.com/mpariente/pystoi
           Args:
               target (string): Name of file to read
               enhanced (string): Name of file to read
               sr (int): sample rate of files
           Returns:
               STOI metric (float)
                   """
    return stoi(target, enhanced, sr)

def read_audio(filename):
    """Read a wavefile and return as numpy array of floats.
            Args:
                filename (string): Name of file to read
            Returns:
                ndarray: audio signal
            """
    try:
        wave_file = SoundFile(filename)
    except:
        # Ensure incorrect error (24 bit) is not generated
        raise Exception(f"Unable to read {filename}.")
    return wave_file.read()

def run_metrics(input_file, save_dir):

    fs = 16000
    enh_file = input_file['enhanced']
    tgt_file = input_file['clean']

    metrics_file = os.path.join(save_dir, f"{input_file['speaker_id']}_{input_file['noise_type']}_{input_file['snr']}_{input_file['file_name']}.pkl")

    # Skip processing with files dont exist or metrics have already been computed
    if ( not os.path.isfile(enh_file) ) or ( not os.path.isfile(tgt_file) ) or ( os.path.isfile(metrics_file)) :
        return

    # Read enhanced signal
    enh = read_audio(enh_file)
    # Read clean/target signal
    clean = read_audio(tgt_file)

    # Check that both files are the same length, otherwise computing the metrics results in an error
    if len(clean) != len(enh):
        raise Exception(
            f"Wav files {enh_file} and {tgt_file} should have the same length"
        )

    # Compute metrics
    m_stoi = compute_stoi(clean, enh, fs)
    m_pesq = compute_pesq(clean, enh, fs)
    m_sisdr = compute_sisdr(clean, enh)

    di_ = {'File name': input_file['file_name'],
          'Noise Type': input_file['noise_type'],
          'Noise SNR':input_file['snr'],
          'SI-SDR': m_sisdr,
          'STOI': m_stoi, 
          'PESQ': m_pesq
          }
    
    save_dict(di_, metrics_file)

def compute_metrics(input_params, save_dir):
    
    futures = []
    ncores = 20
    with ProcessPoolExecutor(max_workers=ncores) as executor:
        for param_ in input_params:
            futures.append(executor.submit(run_metrics, param_, save_dir))
        proc_list = [future.result() for future in tqdm(futures)]

    df_metrics = pd.DataFrame(columns=['File name', 'Noise Type', 'Noise SNR', 'SI-SDR', 'STOI', 'PESQ'])
    
    pkl_files = [f for f in os.listdir(save_dir) if f.endswith('.pkl')]
    
    # Store results in one file
    for this_file in tqdm(pkl_files):
        this_file_path = os.path.join(save_dir, this_file)
        this_res = load_dict(this_file_path)
        
        df_metrics = pd.concat([df_metrics, pd.DataFrame.from_dict({'File name': [this_res['File name']], 'Noise Type': [this_res['Noise Type']],
                                        'Noise SNR': [this_res['Noise SNR']], 'PESQ': [this_res['PESQ']], 'STOI': [this_res['STOI']],
                                        'SI-SDR': [this_res['SI-SDR']]})], ignore_index=True)   

        # remove tmp file
        os.system(f"rm {this_file_path}")


    # Save the DataFrame to a CSV file
    df_metrics.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)

if __name__ == "__main__":
    parser = ArgumentParser(description='Run performance evaluation metrics on the enhanced signals.')
    parser.add_argument("--data_dir", type=str, required=True, help='Directory to the test data.')
    parser.add_argument("--enhanced_dir", type=str, required=True, help='Directory to the enhanced test data.')
    parser.add_argument("--save_dir", type=str, required=True, help='Directory to save the results.')
    args = parser.parse_args()
    
    # load file list and select the target segment to process
    files_list = load_dict(args.data_dir)
    
    input_params = [{'noisy':filename['mix_file'], 'clean':filename['speech_file'], 'file_name':filename['file_name'],
                     'noise_type':filename['noise_type'], 'snr':filename['snr'], 'speaker_id':filename['speaker_id'],
                     'enhanced':f"{args.enhanced_dir}/{filename['speaker_id']}_{filename['noise_type']}_{filename['snr']}_{filename['file_name']}.wav",
                    }
                    for filename in files_list]
    
    compute_metrics(input_params, args.save_dir)
