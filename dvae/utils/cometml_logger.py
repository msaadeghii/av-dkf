"""!
@brief Library for experiment cometml functionality

@author Mostafa Sadeghi {mostafa.sadeghi@inria.fr}
@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import numpy as np
import torch
from asteroid.metrics import get_metrics
import librosa
import matplotlib.pyplot as plt
import librosa.display

class AudioSpecLogger(object):
    def __init__(self, STFT_dict, compute_metrics = False):
        """
        :param STFT_dict: dictionary of the STFT params
        
        """
        
        self.compute_metrics = compute_metrics
        self.fs = STFT_dict['fs']
        self.nfft = STFT_dict['nfft']
        self.hop = STFT_dict['hop']
        self.wlen = STFT_dict['wlen']
        self.win = STFT_dict['win']

    def log_audio_spectrogram(self,
                         input_spec,
                         recon_spec,
                         input_phase,
                         experiment,
                         tag='',
                         step=None,
                         max_batch_items=4):
   
        
        for b_ind in range(min(input_spec.shape[0], max_batch_items)):
            
            this_input_spec = input_spec[b_ind]
            this_recon_spec = recon_spec[b_ind]
            this_input_phase = input_phase[b_ind]
            
            this_input_stft = torch.abs(torch.sqrt(this_input_spec.clone())) * torch.exp(1j * this_input_phase)
            this_recon_stft = torch.abs(torch.sqrt(this_recon_spec.clone())) * torch.exp(1j * this_input_phase)
            
            this_input_wav = torch.istft(this_input_stft, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen, window=torch.from_numpy(self.win).to('cuda'),
                        center=True, normalized=False, onesided=True, length=None, return_complex=False).detach().cpu().numpy() 

            this_recon_wav = torch.istft(this_recon_stft, n_fft=self.nfft, hop_length=self.hop, win_length=self.wlen, window=torch.from_numpy(self.win).to('cuda'),
                        center=True, normalized=False, onesided=True, length=None, return_complex=False).detach().cpu().numpy() 
                                    
            if self.compute_metrics:
                metrics_dict = get_metrics(mix = this_recon_wav, clean = this_input_wav, estimate = this_recon_wav, sample_rate=self.fs, metrics_list=['si_sdr', 'stoi', 'pesq'])
                output_scores = {'SI-SDR':metrics_dict['si_sdr'], 'PESQ':metrics_dict['pesq'], 'STOI':metrics_dict['stoi']}
                experiment.log_metrics(output_scores, prefix='autoencoding_metrics', step=step)
            
            experiment.log_audio(this_input_wav.squeeze(),
                                 sample_rate=self.fs,
                                 file_name=tag+'batch_{}_{}'.format(b_ind+1, 'input'),
                                 metadata=None, overwrite=True,
                                 copy_to_tmp=True, step=step)

            experiment.log_audio(this_recon_wav.squeeze(),
                                 sample_rate=self.fs,
                                 file_name=tag+'batch_{}_{}'.format(b_ind+1, 'recon'),
                                 metadata=None, overwrite=True,
                                 copy_to_tmp=True, step=step)
            
            Spec_in = librosa.amplitude_to_db(torch.abs(this_input_stft).detach().cpu().numpy().squeeze(), ref=np.max)            
            Spec_re = librosa.amplitude_to_db(torch.abs(this_recon_stft).detach().cpu().numpy().squeeze(), ref=np.max)
            
            spec_fig_in, ax = plt.subplots(1,1)
            img_in = librosa.display.specshow(Spec_in, y_axis='log', x_axis='time', ax= ax)

            experiment.log_figure(figure=spec_fig_in, figure_name=tag+'batch_{}_{}'.format(b_ind+1, 'input'), overwrite=True, step=step)
            plt.close(spec_fig_in)
            
            spec_fig_re, ax = plt.subplots(1,1)
            img_re = librosa.display.specshow(Spec_re, y_axis='log', x_axis='time', ax= ax)

            experiment.log_figure(figure=spec_fig_re, figure_name=tag+'batch_{}_{}'.format(b_ind+1, 'recon'), overwrite=True, step=step)
            plt.close(spec_fig_re)
            
def report_se_metrics(res_dic, experiment, tr_step, val_step):
    """Wrapper for cometml metrics report functionality.
    Reports the mean and the std of each metric by inferring the train and the
    val string and it assigns it accordingly.
    Args:
        losses_dict: Python Dict with the following structure:
                     res_dic[loss_name] = {'mean': 0., 'std': 0., 'acc': []}
        experiment:  A cometml experiment object
        tr_step:     The step/epoch index for training
        val_step:     The step/epoch index for validation
    Returns:
        The updated losses_dict with the current mean and std
    """

    for l_name in res_dic:
        values = res_dic[l_name]['acc']
        mean_metric = np.mean(values)
        median_metric = np.median(values)
        std_metric = np.std(values)

        with experiment.validate():
            experiment.log_metric(
                f'{l_name}_mean', mean_metric, step=val_step)
            experiment.log_metric(
                f'{l_name}_median', median_metric, step=val_step)
            experiment.log_metric(
                f'{l_name}_std', std_metric, step=val_step)

        res_dic[l_name]['mean'] = mean_metric
        res_dic[l_name]['median'] = median_metric
        res_dic[l_name]['std'] = std_metric

    return res_dic
