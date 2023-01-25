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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models_dict = {
    "A-VAE": "./saved_model/A-VAE/A-VAE.pt",
    "AV-VAE": "./saved_model/AV-VAE/AV-VAE.pt",
    "A-DKF": "./saved_model/A-DKF/A-DKF.pt",
    "AV-DKF": "./saved_model/AV-DKF/AV-DKF.pt",
}

available_models = list(models_dict.keys())

@st.cache(ttl=None, allow_output_mutation=True, max_entries=3)
def load_pretrained_model(model_path, algo_type = 'peem', verbose = False, niter = 200, save_flg = True, output_dir = "./results"):
    path_model, _ = os.path.split(model_path)
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    se = SpeechEnhancement(saved_model = model_path, output_dir = output_dir, nmf_rank = 8, niter = niter, device = DEVICE, save_flg = save_flg, verbose = verbose)
    return se


st.title("Visualizing Speech Enhancement using Pretrained Generative Models")

speakers_id = ['09F',  '24M',  '26M',  '27M',  '33F',  '40F',  '47M',  '49F', '56M']
noise_types = ['Babble',  'Cafe',  'Car',  'LR', 'Street', 'White']
SNRs = [-5,0,5,10,15]

output_dir = "./results"

speaker_id = st.sidebar.selectbox(
    "Choose Speaker ID",
    speakers_id
)
noise_type = st.sidebar.selectbox(
    "Choose Noise Type",
    noise_types
)
SNR = st.sidebar.selectbox(
    "Choose Noise SNR",
    SNRs
)

def get_data(speaker_id, noise_type, SNR = 0, root_dir = None, fs = 16000):
    if root_dir is None:
        # change this to your own data dir
        root_dir = '/pathTo/corpus/'
    speech_dir = os.path.join(root_dir, 'audio_visual', 'TCD-TIMIT', 'test_data_NTCD', 'clean')
    mix_dir = os.path.join(root_dir, 'audio_visual', 'NTCD-TIMIT-noisy')

    fnames = [x[:-4] for x in os.listdir(os.path.join(mix_dir, noise_type, str(SNR), 'volunteers', speaker_id, 'straightcam')) if x.endswith(".wav")]

    out_dict = {fname: [os.path.join(mix_dir, noise_type, str(SNR), 'volunteers', speaker_id, 'straightcam', fname+'.wav'),
                os.path.join(speech_dir, speaker_id, fname+'.wav'),
                os.path.join(speech_dir, speaker_id, fname+'Raw.npy')] for fname in  fnames}

    return fnames, out_dict

fnames, out_dict = get_data(speaker_id, noise_type, SNR)

fname = st.sidebar.selectbox(
    "Choose Sample",
    fnames
)
mix_file, speech_file, video_file = out_dict[fname]

model_name = st.sidebar.selectbox(
    "Choose the Generative Model",
    available_models
)
model_path = models_dict[model_name]

if model_name in ["A-VAE", "AV-VAE"]:
    algo_types = ["peem", "gpeem"]
else:
    algo_types = ["dpeem", "gdpeem"]

algo_type = st.sidebar.selectbox(
    "Choose Algorithm",
    algo_types
)
niter = int(st.sidebar.text_input('Number of Iterations', value='100'))
show_audio = st.sidebar.checkbox("Show Audios")
show_video = st.sidebar.checkbox("Show Videos")
show_spec = st.sidebar.checkbox("Show Spectrograms")
comp_trace = st.sidebar.checkbox("Compute Trace")
show_trace = st.sidebar.checkbox("Show Trace")


se = load_pretrained_model(model_path, algo_type = algo_type, verbose = False, niter = niter, save_flg = True, output_dir = output_dir)
if "state_dict" not in st.session_state:
    st.session_state.state_dict = {}

def add_to_exp(experience):
    score_sisdr, score_pesq, score_stoi, info = se.run([mix_file, speech_file, video_file, algo_type], tqdm=stqdm, experience_name = model_name, compute_trace = comp_trace)

    save_dir = os.path.join(output_dir, noise_type, str(SNR), speaker_id)
    speech_est_file = os.path.join(save_dir, fname+f'_speech_est_{algo_type}_{model_name}' +'.wav')
    noise_est_file = os.path.join(save_dir,fname+f'_noise_est_{algo_type}_{model_name}' +'.wav')
    mix_speech_file = os.path.join(save_dir,fname+'_mix_norm.wav')
    video_path = os.path.join(save_dir,fname+'_video.avi')
    v_path_s_est_avi = os.path.join(save_dir, fname+f'_speech_est_{algo_type}_{model_name}' +'.avi')
    v_path_s_est = os.path.join(save_dir, fname+f'_speech_est_{algo_type}_{model_name}' +'.mp4')
    v_path_clean_avi = os.path.join(save_dir,fname+f'_clean.avi')
    v_path_clean = os.path.join(save_dir,fname+f'_clean.mp4')
    v_path_mix_avi = os.path.join(save_dir,fname+'_mix_norm.avi')
    v_path_mix = os.path.join(save_dir,fname+'_mix_norm.mp4')
    path_trace = os.path.join(save_dir, fname+f'_trace_{algo_type}_{model_name}' +'.mp4')

    os.system(f"ffmpeg -i {video_path} -i {speech_est_file} -c:v copy -c:a aac {v_path_s_est_avi} -y")
    os.system(f"ffmpeg -y -i {v_path_s_est_avi} -vcodec libx264 {v_path_s_est}")
    os.system(f"ffmpeg -i {video_path} -i {speech_file} -c:v copy -c:a aac {v_path_clean_avi} -y")
    os.system(f"ffmpeg -y -i {v_path_clean_avi} -vcodec libx264 {v_path_clean}")
    os.system(f"ffmpeg -i {video_path} -i {mix_speech_file} -c:v copy -c:a aac {v_path_mix_avi} -y")
    os.system(f"ffmpeg -y -i {v_path_mix_avi} -vcodec libx264 {v_path_mix}")
    if comp_trace:
        os.system(f"ffmpeg -framerate 10 -i temp/frame%03d.png {path_trace}")

    log = 'OUTPUT SI-SDR: {} --- PESQ: {} --- STOI: {}'.format(score_sisdr, score_pesq, score_stoi)

    out_dict = {"speech_est": speech_est_file,
                "noise_est": noise_est_file,
                "mix": mix_speech_file,
                "speech_est_video": v_path_s_est,
                "speech_clean_video": v_path_clean,
                "speech_mix_video": v_path_mix,
                "clean": speech_file,
                "info": info,
                "path_trace": path_trace,
                "log": log,
                "model_type": model_name}

    st.session_state["state_dict"][experience] = out_dict

def add_to_app(experience, num = 0):
    exp_dict = st.session_state["state_dict"][experience]
    exp_title = f'<p style="font-family:Fantasy; color:Gray; font-size: 40px;">Experiment {num}: {exp_dict["model_type"]} </p>'
    st.markdown(exp_title, unsafe_allow_html=True)
    st.write(experience)

    info = exp_dict["info"]
    input_scores = info["input_scores"]
    input_log = 'INPUT SI-SDR: {} --- PESQ: {} --- STOI: {}'.format(input_scores[0], input_scores[1], input_scores[2])
    st.write(input_log)
    st.write(exp_dict["log"])


    if show_spec:
        spec_figure = info["spec_figure"]
        st.pyplot(spec_figure)

    if show_trace:
        if os.path.exists(exp_dict["path_trace"]):
            trace_video = open(exp_dict["path_trace"], 'rb').read()
        st.video(trace_video)

    au1, au2, au3 = st.columns(3)
    au1.subheader("Speech Estimation")
    au2.subheader("Clean Speech")
    au3.subheader("Input Mix File")
    if show_audio:
        speech_est = open(exp_dict["speech_est"], 'rb').read()
        noise_est = open(exp_dict["noise_est"], 'rb').read()
        clean_speech = open(exp_dict["clean"], 'rb').read()
        mix = open(exp_dict["mix"], 'rb').read()
        with au1:
            st.audio(speech_est, format='audio/wav')
        with au2:
            st.audio(clean_speech, format='audio/wav')
        with au3:
            st.audio(mix, format='audio/wav')

    if show_video:
        speech_est_video = open(exp_dict["speech_est_video"], 'rb').read()
        speech_clean_video = open(exp_dict["speech_clean_video"], 'rb').read()
        speech_mix_video = open(exp_dict["speech_mix_video"], 'rb').read()

        with au1:
            st.video(speech_est_video)
        with au2:
            st.video(speech_clean_video)
        with au3:
            st.video(speech_mix_video)


    st.markdown("""---""")



run_cell, clear_cell = st.sidebar.columns(2)
with run_cell:
    if st.sidebar.button("RUN"):
        experience_name = "_".join([str(x) for x in [speaker_id, noise_type, SNR, fname, model_name, algo_type, niter]])
        add_to_exp(experience_name)
with clear_cell:
    if st.sidebar.button("CLEAR"):
        st.session_state.state_dict = {}

app_state = st.session_state["state_dict"]
for num, experience in enumerate(app_state):
    add_to_app(experience, num+1)
