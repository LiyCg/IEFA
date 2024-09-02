from argparse import ArgumentParser
import argparse

def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')



def add_common_options(parser):
    group = parser.add_argument_group('common')
    group.add_argument("--device", default=1, type=int, help="Device id to use. 0: cpu, 1 ~ : gpu")
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=1, type=int, help="Batch size during training.")
    group.add_argument("--wandb", type=bool, default=True, help="Use wandb to log training process")
    group.add_argument("--weight_decay", default=0.0, type=float, help="Optimizer weight decay.")
    group.add_argument("-w", "--num_workers", type=int, default=16, help="dataloader worker size")
    group.add_argument("--sr", type=int, default = '16000', help="audio sample rate")
    group.add_argument("--vtx_dim", type=int, default=28227, help='vtx dim of ict model')
    group.add_argument("--vertice_dim", type=int, default=28227, help='number of vertices - 9408*3 for ict face area')
    group.add_argument("--period", type=int, default=30, help='period in PPE - 30 for mead (o.vocaset)')
    group.add_argument("--fps", default=30, type=int, help="fps 30 for mead")
    group.add_argument("--render_batch", default=120, type=int, help="Batch size for rendering")
    # File name
    group.add_argument("--neutral_vtx_file", type=str, default='ict_M003_front_neutral_1_011_last_fr.npy', help="vtx file of the flame model (pickle)")
    group.add_argument("--wav_file", type=str, default='dataset_m003_wavform.pickle', help="wav file (pickle)")
    group.add_argument("--vtx_file", type=str, default='dataset_m003_vtx.pickle', help="vtx file of the flame model (pickle)")
    group.add_argument("--gemini_file", type=str, default='m003_image_gemini_0117.pickle', help="Image to text caption. Question: Please describe the facial expressions and emotions of the person.")
    # Directory
    #group.add_argument("--root_dir", type=str, default='../../', help="sys path")
    group.add_argument("--root_dir", type=str, default='/input/inyup/TeTEC/faceClip/', help="sys path")
    group.add_argument("--feature_dir", type=str, default='data/feature', help="Path to feature files")
    group.add_argument("--save_vtx_dir", type=str, default='data/test/result/vtx', help="save directory for the output flame parameter")
    group.add_argument("--save_video_dir", type=str, default='data/test/result/qualitative_render/IEFA', help="save directory for the output animation")
    group.add_argument("--template_filepath", type=str, default='render/templates/FLAME_sample.ply', help="FLAME template file path")
    group.add_argument("--image_dir", type=str, default = 'data/mead', help="Path to video folders")


## TODO
# def add_IEFA_options(parser):
    
#     group = parser.add_argument_group('common')
    
#     group.add_argument("--device", default=1, type=int, help="Device id to use. 0: cpu, 1 ~ : gpu")
#     group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
#     group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
#     group.add_argument("--batch_size", default=1, type=int, help="Batch size during training.")

def add_disentanglement(parser): # Stage 1
    group = parser.add_argument_group('disentanglement')
    group.add_argument("--model_num", type=str, default='ict_default', help="Model name.")
    group.add_argument("-e", "--epochs", type=int, default=10000, help="number of epochs")
    group.add_argument("--lr", default=1e-5, type=float, help="Learning rate.")

    group.add_argument("--feature_dim", type=int, default=1024, help='latent space dimension')
    group.add_argument("--num_heads", default=4, type=int, help="Number of heads (transformer)")
    group.add_argument("--num_layers", default=4, type=int, help="Number of layers")

    group.add_argument("--triplet_margin", type=int, default=1, help='Margin of the triplet loss')
    # loss weight
    group.add_argument("--w_cross", default=10000, type=float, help="Loss weight for cross recon")
    group.add_argument("--w_self", default=10000, type=float, help="Loss weight for self recon")
    group.add_argument("--w_con", default=0.001, type=float, help="Loss weight for content embedding")
    group.add_argument("--w_tpl", default=0.001, type=float, help="Loss weight for expression triplet loss")
    # dir
    group.add_argument("--save_dir", type=str, default='/source/inyup/TeTEC/faceClip/src/disentanglement/ckpts/disentanglement', help="Path to save checkpoints and results.")
    group.add_argument("--save_tb_dir", type=str, default='src/disentanglement/ckpts/disentanglement/tensorboard', help="Path to save logs.")
    group.add_argument("--vtx_dtw_path", type=str, default='data/feature/dataset_m003_vtx_dtw.pickle', help="time warped vtx animation data path")

def add_a2s_options(parser): # Stage 2
    group = parser.add_argument_group('a2s_model')
    group.add_argument("--model_name", type=str, default = '23', help="Model name.")
    group.add_argument("-e", "--epochs", type=int, default=2000, help="number of epochs")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")

    group.add_argument("--dataset", type=str, default = 'mead', help="dataset name: mead or ...")
    group.add_argument("--num_layers", default=4, type=int, help="Number of layers")
    group.add_argument("--num_heads", default=4, type=int, help="Number of heads (transformer)")
    group.add_argument("--triplet_margin", type=int, default=1, help='Margin of the triplet loss')

    group.add_argument("--audio_dim", type=int, default=1024, help='64 / wav2vec dim')
    group.add_argument("--clip_dim", type=int, default=1024, help='512 -> change after model 18')
    group.add_argument("--feature_dim", type=int, default=1024, help='same with the feature dim of the autoencoder')
    #group.add_argument("--w_emb", default=0.00001, type=float, help="Loss weight for content and expression embedding features")

    # Directory
    group.add_argument("--save_tb_dir", type=str, default = 'ckpts/a2s/tensorboard', help="Path to save logs.")
    group.add_argument("--save_dir", type=str, default='ckpts/a2s', help="Path to save checkpoints and results.")
    group.add_argument("--emotion_words", type=str, default='emotion_words.pickle', help="Emotion words for text description")
    group.add_argument("--autoencoder_path", type=str, default='ckpts/disentanglement/ict_default.pth', help="Pre-trained autoencoder in Stage 1(Disentanglement)")

def add_test_options(parser):
    group = parser.add_argument_group('test')
    group.add_argument("--eval_save_dir", type=str, default = 'data/test/result/qualitative_render')
    group.add_argument("--test_wav_dir", type=str, default = 'data/test/audio/userstudy', help="test wav file path")
    group.add_argument("--test_audio", type=str, default = 'm03-angry-level_3-003.wav', help="test wav file path + file name")
    #group.add_argument("--text", type=str, default = 'His happiness is evident in the sparkle of his eyes and the warmth of his smile.', help="expression condition (text description)")
    # He has his eyes wide open and a big smile on his face.
    # He looks like he is about to cry.
    group.add_argument("--text", type=str, default = "neutral", help="expression condition (text description)")
    # group.add_argument("--test_con_vtx", type=str, default = 'data/test/result/vtx/M003_front_angry_3_003_dtw.npy', help="test vtx file for the input of content encoder")
    # group.add_argument("--test_exp_vtx", type=str, default = 'data/test/result/vtx/M003_front_angry_3_003_dtw.npy', help="test vtx file for the input of expression encoder")

# ---------------------------------------------------------------------------------- #

# Content - Expression Disentanglement (Stage 1)
def disentangle_args():
    parser = ArgumentParser()
    add_common_options(parser)
    add_disentanglement(parser)
    add_test_options(parser)
    opt = parser.parse_args()
    return opt

# Audio 2 Speech (Stage 2)
def a2s_args():
    parser = ArgumentParser()
    add_common_options(parser)
    add_a2s_options(parser)
    add_test_options(parser)
    opt = parser.parse_args()
    return opt

def IEFA_args():
    parser = ArgumentParser()
    add_common_options(parser)
    # add_IEFA_options(parser)
    opt = parser.parse_args()
    return opt