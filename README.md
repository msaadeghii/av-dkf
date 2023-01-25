# Audio-visual Deep Kalman Filter (AV-DKF)

This repository contains the codes for the audio-visual speech enhancement (AVSE) framework based on Deep Kalman Filter (DKF), which is proposed in the following paper:

> [1] A. Golmakani, M. Sadeghi, and R. Serizel, [Audio-visual Speech Enhancement with a Deep Kalman Filter Generative Model](https://arxiv.org/abs/2211.00988), October 2022.

We have largely used the [DVAE repository](https://github.com/XiaoyuBIE1994/DVAE) to write these codes.

## Getting started

To get started with the codes, you will need to first `train` a model and then evaluate it by either `speech analysis-resynthesis` or `speech enhancement`. To know different directories and main functions in this repository, please consult the following.

### Training

Assuming that you have already cloned this repository and are in the `av-dkf` directory:

1. All the VAE models (audio-only, video-only, audio-visual) that process data frames independently, i.e. without taking into account a temporal dynamic model, are provided in `./dvae/model/vae.py`. Dynamical VAE models, mainly for the audio-only case (for now, audio-visual is available for only the DKF model), are provided as separate functions in `./dvae/model/`. The associated training loss functions are defined inside each VAE class.

2. Dataset models are provided in `./dvae/utils/speech_dataset.py`, where `SpeechDatasetFrames` is to load data frames independently, i.e. for simple, non-dynamic VAE models, while `SpeechDatasetSequences`, `SpeechSequencesRandom` and `SpeechSequencesFull` are to be used for Dynamical VAE models.

3. Network configurations, STFT parameter settings, training parameters, etc. are summarized in some dedicated config file that can be found in `./example_configuration/`. For instance, `cfg_vae_A_VAE.ini` contains configurations for the A-VAE model (audio-only, non-dynamical VAE).

4. Training and evaluation (speech analysis-resynthesis) functions are provided in `./dvae/learning_algo.py`. Make sure that you set the right dataset loader inside this function (see item 2).

5. The main training bash script is `./dvae/training/run_oar_NTCD.sh`.

### Pretrained models

You can find some pretrained VAE models in the `saved_model` directory. The provided models are `A-VAE` & `AV-VAE` for non-dynamical models, and `A-DKF` & `AV-DKF` for dynamical (DKF) models.

### Evaluation

As said earlier, you can evaluate the performance of your trained model via either `speech analysis-resynthesis` or `speech enhancement`. In the former case, the input speech is first encoded by the trained VAE encoder to get the corresponding latent codes, and then an estimate of the input speech is reconstructed in the output of the decoder using the latent codes as the input. The quality of the reconstructed speech is then compared with the input speech data. The corresponding function is the `generate` method within `./dvae/learning_algo.py`. You can find the script `./test_speech_analysis_resynthesis.py` useful to perform this task.

The speech enhancement scripts are provided in `./dvae/SE/`, where `SE_algorithms.py` is for non-dynamical VAE models mainly (but it also contains codes for the DKF models), while `DSE_algorithms.py` is for Dynamical VAE models. The script `Test_SE.py` runs speech enhancement for a single test sample from the TCD-TIMIT dataset.

The working enhancement algorithms are `peem` & `gpeem` for non-dynamical VAE models, and `dpeem` & `gdpeem` for the dynamical VAE models. All these methods are based on finding the mode of the posterior distribution in the expectation step using the Adam optimizer. Here, "g" stands for the gradient-based gain update method proposed in our paper [1].

## Speech enhancement demo

The Jupyter notebook `SE_demo.ipynb` provides a simple demo of applying different VAE models with different enhancement algorithms to the sample data provided in `./data`.

## Streamlit application

You can set up a visual interface to run different speech enhancement models on TCD-TIMIT test data. For that, you should first make sure that `streamlit` is already installed in your virtual environment. Then, within the main directory of `av-dkf`, you simply run `streamlit run app/streamlit_app.py`, and then open `http://localhost:8501/` in your web browser, e.g., Chrome. You should also modify the data paths inside the scripts provided in `./app/pages`

You can then interact with the app and choose the target speech signal, enhancement algorithm, and other desired parameters.

## Useful resources

You can find some good resources on the DVAE models [here](https://dynamicalvae.github.io/).

[This repository](https://github.com/XiaoyuBIE1994/DVAE_SE) provides the codes to implement a variational expectation-maximization (VEM) approach to audio-only speech enhancement based on the DVAE models.

## Contact

The main contributors are Ali Golmakani (golmakani77[at]yahoo[dot]com) and Mostafa Sadeghi (mostafa[dot]sadeghi[at]inria[dot]fr).
