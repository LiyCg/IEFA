import torch
import os
import pickle
import sys
from argparse import ArgumentParser
import argparse
from MEO.llm.motion_io import Motion_DB
from MEO.openai_wrapper import read_progprompt_0, read_progprompt, get_incontext, query_model
from disentanglement.train import Runner
from disentanglement.test import direct_decoding
from parser_util import *
sys.path.append("/input/inyup/TeTEC/faceClip/data/")
sys.path.append('/disentanglement/')
from livelink_MEAD.util import bshp_2_vtx


db = Motion_DB()

def run_pipeline():
            
    prompt_sequence = []
    
    ## TODO 
    ## - (DONE) turned this into run each found methods one by one
    ## - check if this actually saves output_anim_seq in the DB
    ## - make the save
    read_progprompt_0() # this reads 'FacialMotion' class
    read_progprompt("")
    get_incontext()

    # import pdb;pdb.set_trace()
    user_input = input("You: ")
    
    print("Chatbot:", user_input)
    prompt = user_input

    prompt_sequence.append("# " + prompt + "\n")
    error_prompt_sequence = prompt_sequence

    c, r = query_model(prompt_sequence, error_prompt_sequence, 0)
    prompt_sequence.append(c)
    print(c)
    


if __name__ == "__main__":

    hparams = IEFA_args()
    device = torch.device("cuda:" + str(hparams.device - 1)) if hparams.device > 0 else torch.device("cpu")
    
    print("---------------- Running IEFA editing pipeline ------------------")
    model_path = f"{os.path.join(hparams.model_root_dir, hparams.save_dir)}/{hparams.model_num}.pth"
    audio_path = f"" # TODO : input in the exact audio file path you want to intput
    runner = Runner(hparams)
    
    runner.autoencoder.load_state_dict(torch.load(model_path, map_location='cuda:0')['autoencoder_state_dict'])
    runner.autoencoder.eval()
    
    ## loading direct neutral vtx animation sequence from captured data. Later will be replaced /w CodeTalker's prediction output
    f = open(os.path.join(hparams.con_data_root_dir, hparams.vtx_dtw_path), 'rb')
    vtx = pickle.load(f)
    con_vtx_anim_seq = vtx["0"]["neutral_1_M003_front_neutral_front_neutral_1_001"]
    
    motion_id = 0
    while True:
        run_pipeline() # 1. queries GPT / 2. run GPT created code 
        motion_id += 1
        motion_info = db.load_motion(f"motion_{motion_id}", return_dict=True)
        exp_bshp_anim_seq = motion_info["output_animation_seq"]
        exp_vtx_anim_seq = bshp_2_vtx(exp_bshp_anim_seq)
        pred_vtx = direct_decoding(hparams=hparams,
                        motion_num = motion_id, 
                        con_vtx_anim_seq=con_vtx_anim_seq, 
                        exp_vtx_anim_seq=exp_vtx_anim_seq,
                        audio_path=audio_path, # for mux
                        runner=runner, 
                        render=True)
