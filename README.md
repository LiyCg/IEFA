# Interactive Editing of Facial Animation with Natural Language

### Inyup Lee


<!-- # Audio-Driven Speech Animation with Text-Guided Expression

### Sunjin Jung
* Avatar Intern (2023.10.30 ~ 2024.01.26)

---
Given an audio and text description, our approach can generate expressive speech animation.
Our system consists of two stages.
The goal of the first stage is to separate content features and expression features in facial animation.
In the second stage, the final speech animation is generated from an audio and text description.
In this stage, the mouth movements and facial expressions are produced from the audio and text, respectively.

## Installation

We train and test our system based on Python 3.7 and Pytorch 1.12.1. 

1. To install the dependencies, run:

```
pip install -r requirements.txt
```

2. Download the pre-trained models and data from the following path:
  - Models: `irteamsu@cdevgpu27.voice.nfra.io:/home1/irteamsu/TTA/codes/sj/Project/faceClip/ckpts` 
  - Data: `irteamsu@cdevgpu27.voice.nfra.io:/home1/irteamsu/TTA/codes/sj/Project/faceClip/data/feature`



## Stage 1
![visualization_s1](./data/figure/stage1.png)


### Train

- Run the code to train the auto-encoder. You can change the model num and choose the device number (0 for CPU, 1 and above for GPU).
```
python disentanglement/train.py --model_num '1' --device 1
```

### Test

- Run the code to test the trained auto-encoder. 
You can change the input facial animations (vtx) for the content and expression features, respectively.
The audio file is not utilized for this model; instead, it is employed for rendering the result animation along with the audio.
Therefore, we recommend using the audio file that corresponds to the facial animation for the content feature.

```
python disentanglement/test.py --model_num '20_e2000' --device 1 --test_audio 'M003_front_angry_3_003.wav' --test_con_vtx '../../data/test/result/vtx/M003_front_angry_3_003_dtw.npy' --test_exp_vtx '../../data/test/result/vtx/M003_front_angry_3_003_dtw.npy'
```



## Stage 2
![visualization_s2](./data/figure/stage2.png)

### Train

- Run the training code with the pre-trained auto-encoder (`20_e2000.pth`). You can change the model name and choose the device number (0 for CPU, 1 and above for GPU).
```
python audio2speech/a2s_train.py --model_name '1' --device 1 --autoencoder_path '../../ckpts/disentanglement/20_e2000.pth'
```

### Test

- Run the testing code with the input audio and text description:
```
python disentanglement/train.py --model_num '23' --device 1 --test_audio 'M003_front_angry_3_003.wav' --text 'He looks like he is about to cry.'
```
Place the input audio file in the `/data/test/audio` directory.
Alternatively, you can specify the test audio file path by using the `--test_wav_dir` arguments. -->