import os
import sys
sys.path.extend("../../")

import numpy as np
import datetime
import torch
from dvae import LearningAlgorithm

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
def load_model(model_path, verbose=False, output_dir = "./results"):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    path, fname = os.path.split(model_path)
    cfg_file = os.path.join(path, 'config.ini')
    learning_algo = LearningAlgorithm(config_file=cfg_file)
    # Load model state
    learning_algo.model.load_state_dict(torch.load(model_path, map_location=learning_algo.device))
    learning_algo.model.eval()
    return learning_algo


st.title("Visualizing Speech Sythesizing")

speakers_id = ['09F',  '24M',  '26M',  '27M',  '33F',  '40F',  '47M',  '49F', '56M']
output_dir = "./speech_synthesize"

speaker_id = st.sidebar.selectbox(
    "Choose Speaker ID",
    speakers_id
)

def get_data(speaker_id, root_dir = None, fs = 16000):
    if root_dir is None:
        root_dir = '/pathTo/corpus/'
    speech_dir = os.path.join(root_dir, 'audio_visual', 'TCD-TIMIT', 'test_data_NTCD', 'clean')

    fnames = [x[:-4] for x in os.listdir(os.path.join(speech_dir, speaker_id)) if x.endswith(".wav")]

    out_dict = {fname: [os.path.join(speech_dir, speaker_id, fname+'.wav'),
                os.path.join(speech_dir, speaker_id, fname+'Raw.npy')] for fname in  fnames}

    return fnames, out_dict

fnames, out_dict = get_data(speaker_id)

fname = st.sidebar.selectbox(
    "Choose Sample",
    fnames
)
speech_file, video_file = out_dict[fname]

model_name = st.sidebar.selectbox(
    "Choose the Generative Model",
    available_models
)
model_path = models_dict[model_name]

# niter = int(st.sidebar.text_input('Number of Iterations', value='100'))
show_audio = st.sidebar.checkbox("Show Audios")
show_video = st.sidebar.checkbox("Show Videos")
show_spec = st.sidebar.checkbox("Show Spectrograms")

algo = load_model(model_path)



if "state_dict_syn" not in st.session_state:
    st.session_state.state_dict_syn = {}

def add_to_exp(experience):
    save_dir = os.path.join(output_dir, speaker_id)
    speech_est_file = os.path.join(save_dir, fname+f'_speech_est_{model_name}' +'.wav')

    score_isdr, score_pesq, score_stoi, spm = algo.generate(audio_orig = speech_file, audio_recon = speech_est_file, save_flag = True, denoise = False, target_snr = None, seed = 666, model_type = algo.vae_mode, save_video = True, trim_s = False)

    mozz
    score_sisdr, score_pesq, score_stoi, info = se.run([mix_file, speech_file, video_file, algo_type], tqdm=stqdm)

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

    os.system(f"ffmpeg -i {video_path} -i {speech_est_file} -c:v copy -c:a aac {v_path_s_est_avi} -y")
    os.system(f"ffmpeg -y -i {v_path_s_est_avi} -vcodec libx264 {v_path_s_est}")
    os.system(f"ffmpeg -i {video_path} -i {speech_file} -c:v copy -c:a aac {v_path_clean_avi} -y")
    os.system(f"ffmpeg -y -i {v_path_clean_avi} -vcodec libx264 {v_path_clean}")
    os.system(f"ffmpeg -i {video_path} -i {mix_speech_file} -c:v copy -c:a aac {v_path_mix_avi} -y")
    os.system(f"ffmpeg -y -i {v_path_mix_avi} -vcodec libx264 {v_path_mix}")

    log = 'OUTPUT SI-SDR: {} --- PESQ: {} --- STOI: {}'.format(score_sisdr, score_pesq, score_stoi)

    out_dict = {"speech_est": speech_est_file,
                "noise_est": noise_est_file,
                "mix": mix_speech_file,
                "speech_est_video": v_path_s_est,
                "speech_clean_video": v_path_clean,
                "speech_mix_video": v_path_mix,
                "clean": speech_file,
                "info": info,
                "log": log}

    st.session_state["state_dict"][experience] = out_dict

def add_to_app(experience):
    exp_dict = st.session_state["state_dict"][experience]
    st.write(experience)
    info = exp_dict["info"]
    st.write(info["input_scores"])
    st.write(exp_dict["log"])


    if show_spec:
        m1 = info["spec_noisy_input"]
        m2 = info["spec_clean_input"]
        m3 = info["spec_noise_output"]
        m4 = info["spec_clean_output"]
        up1 = np.concatenate([m1, np.ones(shape = [m1.shape[0], 10]), m3], axis = 1)
        up2 = np.concatenate([m2, np.ones(shape = [m2.shape[0], 10]), m4], axis = 1)
        specs_image = np.concatenate([up1, np.ones(shape = [10, up1.shape[1]]), up2], axis = 0)
        st.image(specs_image)

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
        experience_name = "_".join([str(x) for x in [speaker_id, fname, model_name]])
        add_to_exp(experience_name)
with clear_cell:
    if st.sidebar.button("CLEAR"):
        st.session_state.state_dict_syn = {}

app_state = st.session_state["state_dict_syn"]
for experience in app_state:
    add_to_app(experience)
