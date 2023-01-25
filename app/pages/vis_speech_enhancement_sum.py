import os
import sys
sys.path.extend("../../")

import numpy as np
import datetime
import torch
from dvae.SE.speech_enhancement import SpeechEnhancement

import streamlit as st
from tqdm import tqdm
from stqdm import stqdm
import plotly.express as px

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models_dict = {
    "A-VAE": "./saved_model/A-VAE/A-VAE.pt",
    "AV-VAE": "./saved_model/AV-VAE/AV-VAE.pt",
    "A-DKF": "./saved_model/A-DKF/A-DKF.pt",
    "AV-DKF": "./saved_model/AV-DKF/AV-DKF.pt",
}

available_models = list(models_dict.keys())

@st.cache(ttl=None, allow_output_mutation=True, max_entries=3)
def load_pretrained_model(model_path, algo_type = 'peem', verbose = False, niter = 200, save_flg = True, output_dir = "./results", experience_name = None):
    path_model, _ = os.path.split(model_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    se = SpeechEnhancement(saved_model = model_path, output_dir = os.path.join(output_dir, ".."), nmf_rank = 8, niter = niter, device = DEVICE, save_flg = save_flg, verbose = verbose)
    return se


st.title("Visualizing Speech Enhancement using Pretrained Generative Models")

speakers_id = ['09F',  '24M',  '26M',  '27M',  '33F',  '40F',  '47M',  '49F', '56M']
noise_types = ['Babble',  'Cafe',  'Car',  'LR', 'Street', 'White']
SNRs = [-10,-5,0,5,10]
algo_types = ["peem", "dpeem", "gpeem", "gdpeem"]
output_dir = "./results/SumExperiences"

exp_name = st.sidebar.text_input('Experience Name')

speaker_id = st.sidebar.selectbox(
    "Choose Speaker ID",
    speakers_id
)
iterate_speaker_id = [speaker_id] if not st.sidebar.checkbox("Iterate Over Speaker ID") else speakers_id
noise_type = st.sidebar.selectbox(
    "Choose Noise Type",
    noise_types
)
iterate_noise_type = [noise_type] if not st.sidebar.checkbox("Iterate Over Noise Type") else noise_types
SNR = st.sidebar.selectbox(
    "Choose Noise SNR",
    SNRs
)
iterate_SNR = [SNR] if not st.sidebar.checkbox("Iterate Over SNR") else SNRs


def get_data(speaker_id, noise_type, SNR = 0, root_dir = None, fs = 16000):
    if root_dir is None:
        root_dir = '/pathTo/corpus/'
    speech_dir = os.path.join(root_dir, 'audio_visual', 'TCD-TIMIT', 'test_data_NTCD', 'clean')
    mix_dir = os.path.join(root_dir, 'audio_visual', 'NTCD-TIMIT-noisy')

    fnames = [x[:-4] for x in os.listdir(os.path.join(mix_dir, noise_type, str(SNR), 'volunteers', speaker_id, 'straightcam')) if x.endswith(".wav")]

    out_dict = {fname: [os.path.join(mix_dir, noise_type, str(SNR), 'volunteers', speaker_id, 'straightcam', fname+'.wav'),
                os.path.join(speech_dir, speaker_id, fname+'.wav'),
                os.path.join(speech_dir, speaker_id, fname+'Raw.npy')] for fname in  fnames}

    return fnames, out_dict

model_name = st.sidebar.selectbox(
    "Choose the Generative Model",
    available_models
)
model_path = models_dict[model_name]

algo_type = st.sidebar.selectbox(
    "Choose Algorithm",
    algo_types
)
niter = int(st.sidebar.text_input('Number of Iterations', value='100'))
limit = int(st.sidebar.text_input('Limit Samples', value='10'))
show_plots = st.sidebar.checkbox("Show Plots")

se = load_pretrained_model(model_path, algo_type = algo_type, verbose = False, niter = niter, save_flg = True, output_dir = output_dir)
if "state_dict" not in st.session_state:
    st.session_state.state_dict = {}

def compute_average_scores(list_scores, list_samples):
    sisdr_mean = np.mean([x[0] for x in list_scores])
    pesq_mean = np.mean([x[1] for x in list_scores])
    stoi_mean = np.mean([x[2] for x in list_scores])
    return sisdr_mean, pesq_mean, stoi_mean

def compute_average_scores_input(list_scores, list_samples):
    sisdr_mean = np.mean([x[-1]["input_scores"][0] for x in list_scores])
    pesq_mean = np.mean([x[-1]["input_scores"][1] for x in list_scores])
    stoi_mean = np.mean([x[-1]["input_scores"][2] for x in list_scores])
    return sisdr_mean, pesq_mean, stoi_mean

def save(experience, list_scores, list_samples):
    save_dir = os.path.join(output_dir, experience)
    if os.path.exists(save_dir):
        kjdhfdkdsjfhdsjf
    else:
        os.makedirs(save_dir)

    np.savez(os.path.join(save_dir, "results.npz"), list_scores = list_scores, list_samples = list_samples)


def load(experience):
    save_dir = os.path.join(output_dir, experience)
    if not os.path.exists(save_dir):
        asdaskdhaslkdjljd
    loaded = np.load(os.path.join(save_dir, "results.npz"), allow_pickle=True)
    list_scores = loaded["list_scores"]
    list_samples = loaded["list_samples"]

    out_dict = {"scores": list_scores,
                "samples": list_samples}

    return out_dict


def add_to_exp(experience):
    list_scores = []
    list_samples = []
    total_iters = len(iterate_speaker_id) * len(iterate_noise_type) * len(iterate_SNR) * limit
    pbar = stqdm(total=total_iters)
    for sid in iterate_speaker_id:
        for nt in iterate_noise_type:
            for snr in iterate_SNR:
                fnames, out_dict = get_data(sid, nt, snr)

                for fname in fnames[:limit]:

                    mix_file, speech_file, video_file = out_dict[fname]
                    score_sisdr, score_pesq, score_stoi, info = se.run([mix_file, speech_file, video_file, algo_type],
                                                                       tqdm=tqdm, experience_name = model_name)
                    list_scores.append([score_sisdr, score_pesq, score_stoi, info])
                    list_samples.append({"speaker_id": sid,
                                         "noise_type": nt,
                                         "SNR": snr,
                                         "mix_file": mix_file,
                                         "speech_file": speech_file,
                                         "video_file": video_file})
                    pbar.update(1)
    pbar.close()




    save(experience, list_scores, list_samples)
    out_dict = load(experience)
    st.session_state["state_dict"][experience] = out_dict

def add_to_app(experience, num = 0):
    exp_title = f'<p style="font-family:Fantasy; color:Gray; font-size: 40px;">Experiment {num}: {experience} </p>'
    st.markdown(exp_title, unsafe_allow_html=True)

    exp_dict = st.session_state["state_dict"][experience]
    list_scores = exp_dict["scores"]
    list_samples = exp_dict["samples"]
    sisdr_mean, pesq_mean, stoi_mean = compute_average_scores(list_scores, list_samples)
    sisdr_mean_input, pesq_mean_input, stoi_mean_input = compute_average_scores_input(list_scores, list_samples)

    sdr_inputs = [x[-1]["input_scores"][0] for x in list_scores]
    pesq_inputs = [x[-1]["input_scores"][1] for x in list_scores]
    stoi_inputs = [x[-1]["input_scores"][2] for x in list_scores]
    sdr_outputs = [x[0] for x in list_scores]
    pesq_outputs = [x[1] for x in list_scores]
    stoi_outputs = [x[2] for x in list_scores]

    sdr_fig = px.scatter(x=sdr_inputs, y=sdr_outputs)
    sdr_fig.update_layout(width=250, height=250, yaxis_title=None, xaxis_title=None,
                         margin=dict(l=10, r=10, t=10, b=10))

    pesq_fig = px.scatter(x=pesq_inputs, y=pesq_outputs)
    pesq_fig.update_layout(width=250, height=250, yaxis_title=None, xaxis_title=None,
                          margin=dict(l=10, r=10, t=10, b=10))

    stoi_fig = px.scatter(x=stoi_inputs, y=stoi_outputs)
    stoi_fig.update_layout(width=250, height=250, yaxis_title=None, xaxis_title=None,
                          margin=dict(l=10, r=10, t=10, b=10))

    au1, au2, au3 = st.columns(3)
    au1.subheader("SDR")
    au2.subheader("PESQ")
    au3.subheader("STOI")
    with au1:
        st.write("Mean Input SDR", sisdr_mean_input)
        st.write("Mean Output SDR", sisdr_mean)
        if show_plots:
            st.plotly_chart(sdr_fig)
    with au2:
        st.write("Mean Input PESQ", pesq_mean_input)
        st.write("Mean Output PESQ", pesq_mean)
        if show_plots:
            st.plotly_chart(pesq_fig)
    with au3:
        st.write("Mean Input STOI", stoi_mean_input)
        st.write("Mean Output STOI", stoi_mean)
        if show_plots:
            st.plotly_chart(stoi_fig)

    st.markdown("""---""")



run_cell, clear_cell = st.sidebar.columns(2)
with run_cell:
    if st.sidebar.button("RUN"):
        add_to_exp(exp_name)
with clear_cell:
    if st.sidebar.button("CLEAR"):
        st.session_state.state_dict = {}

list_experiences = os.listdir(output_dir)
add_exp = st.multiselect("Add Experiment", list_experiences)
if st.button("ADD"):
    for exp in add_exp:
        out_dict = load(exp)
        st.session_state["state_dict"][exp] = out_dict

app_state = st.session_state["state_dict"]
for num, experience in enumerate(app_state):
    add_to_app(experience, num+1)
