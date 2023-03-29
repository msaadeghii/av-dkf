# Audio-visual Deep Kalman Filter (AV-DKF)

This repository contains the codes for the audio-visual speech enhancement (AVSE) framework based on Deep Kalman Filter (DKF) [1], as well as the fast & efficient speech enhancement method proposed in [2].

We have used the [DVAE repository](https://github.com/XiaoyuBIE1994/DVAE) to write these codes.

## Table of contents

  - [Getting started](#getting-started)
  - [Training](#training)
  - [Pretrained models](#pretrained-models)
  - [Evaluation](#evaluation)
  - [Speech enhancement demo](#speech-enhancement-demo)
  - [Streamlit application](#streamlit-application)
  - [Useful resources](#useful-resources)
  - [Contacts](#contacts)
  - [References](#references)
  
## Getting started

To get started with the codes, you need to clone this repository and install required libraries:

```shell
# Clone repository
git clone https://github.com/msaadeghii/av-dkf.git
cd av-dkf

# Create & activate a virtual environment with conda, see https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
conda create --name avse python=3.8
conda activate avse

# Install PyTorch
pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102

# Install OpenCV
pip install opencv-python
```

You may also need to install some additional packages. There are some pre-trained models that you can use. To know different directories and main functions in this repository, please consult the following. 

You can monitor model training using [Comet ML](https://www.comet.com/). To set up an account and get your Comet ML `API key`, follow the instructions provided [here](https://www.comet.com/docs/v2/guides/getting-started/quickstart/). Once you get your API key, insert it in the config files loated in `configs`.

**Note:** You should download a [pretrained lipreading model](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks#model-zoo) for visual feature extraction, and put it in `lipreading/data`.

## Training

Assuming that you have already cloned this repository and are in the `av-dkf` directory:

1. All the VAE models (audio-only, video-only, audio-visual) that process data frames independently, i.e. without taking into account a temporal dynamic model, are provided in `dvae/model/vae.py`. Dynamical VAE models, mainly for the audio-only case (for now, audio-visual is available for only the DKF model), are provided as separate functions in `dvae/model`. The associated training loss functions are defined inside each VAE class.

2. Dataset models are provided in `dvae/utils/speech_dataset.py`, where `SpeechDatasetFrames` is to load data frames independently, i.e. for simple, non-dynamic VAE models, while `SpeechDatasetSequences` and `SpeechSequencesFull` are to be used for Dynamical VAE models.

3. Network configurations, STFT parameter settings, training parameters, etc. are summarized in some dedicated config file that can be found in `configs`. For instance, `cfg_A_VAE.ini` contains configurations for the A-VAE model (audio-only, non-dynamical VAE).

4. Training and evaluation (speech analysis-resynthesis) functions are provided in `dvae/learning_algo.py`. Make sure that you set the right dataset loader inside this function (see item 2).

5. The main training script is `train.sh`.

## Pretrained models

You can find some pretrained VAE models in the `pretrained_models` directory. The provided models are `A-VAE` & `AV-VAE` for non-dynamical models, and `A-DKF` & `AV-DKF` for dynamical (DKF) models, which have been trained on the [NTCD-TIMIT](https://zenodo.org/record/260228) dataset.

## Evaluation

As said earlier, you can evaluate the performance of your trained model via either `speech analysis-resynthesis` or `speech enhancement`. In the former case, the input speech is first encoded by the trained VAE encoder to get the corresponding latent codes, and then an estimate of the input speech is reconstructed in the output of the decoder using the latent codes as the input. The quality of the reconstructed speech is then compared with the input speech data. The corresponding function is the `generate` method within `dvae/learning_algo.py`. You can find the script `test_speech_analysis_resynthesis.py` useful to perform this task.

The speech enhancement scripts are provided in `dvae/SE`. The script `Test_SE.py` runs speech enhancement for a single test sample from the [NTCD-TIMIT](https://zenodo.org/record/260228) dataset.

The working enhancement algorithms are `peem` & `gpeem` for non-dynamical VAE models, and `dpeem` & `gdpeem` for the dynamical VAE models. All these methods are based on finding the mode of the posterior distribution in the expectation step using the Adam optimizer. Here, "g" stands for the gradient-based gain update method proposed in our paper [1].

> The speech enhancement algorithm proposed in [2], which is called Langevin Dynamics Expectation Maximization (LDEM), can be found inside `dvae/SE/SE_algorithms`. You can also try it in the demo notebook (see below).

## Speech enhancement demo

The Jupyter notebook `SE_demo.ipynb` provides a simple demo of applying different VAE models with different enhancement algorithms to the sample data provided in `data`.

## Streamlit application

You can set up a visual interface to run different speech enhancement models on TCD-TIMIT test data. For that, you should first make sure that `streamlit` is already installed in your virtual environment. Then, within the main directory of `av-dkf`, you simply run `streamlit run app/streamlit_app.py`, and then open `http://localhost:8501/` in your web browser, e.g., Chrome. You should also modify the data paths inside the scripts provided in `app/pages`

You can then interact with the app and choose the target speech signal, enhancement algorithm, and other desired parameters.

## Useful resources

You can find some good resources on the DVAE models [here](https://dynamicalvae.github.io/).

[This repository](https://github.com/XiaoyuBIE1994/DVAE_SE) provides the codes to implement a variational expectation-maximization (VEM) approach to audio-only speech enhancement based on the DVAE models.

## Contacts

- [Mostafa Sadeghi](https://msaadeghii.github.io/) (mostafa[dot]sadeghi[at]inria[dot]fr).
- Ali Golmakani (golmakani77[at]yahoo[dot]com)

## References

[1] A. Golmakani, M. Sadeghi, and R. Serizel, [Audio-visual Speech Enhancement with a Deep Kalman Filter Generative Model](https://arxiv.org/abs/2211.00988), in IEEE International Conference on Acoustics Speech and Signal Processing (ICASSP), Rhodes island, June 2023.

[2] M. Sadeghi and R. Serizel, [Fast and Efficient Speech Enhancement with Variational Autoencoders](https://arxiv.org/abs/2211.02728), in IEEE International Conference on Acoustics Speech and Signal Processing (ICASSP), Rhodes island, June 2023.