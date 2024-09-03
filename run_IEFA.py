import torch
import os
import pickle
import datetime
import sys
from argparse import ArgumentParser
import argparse
from src.FaceMEO.llm.motion_io import Motion_DB
from src.FaceMEO.openai_wrapper import read_progprompt_0, read_progprompt, get_incontext, query_model
from src.disentanglement.train import Runner
from src.disentanglement.test import direct_decoding
from parser_util import IEFA_args

from data.livelink_MEAD.util import bshp_2_vtx

db = Motion_DB()

def run_pipeline(prompt_sequence, context):
            
    ## TODO 
    ## - (DONE) turned this into run each found methods one by one
    ## - check if this actually saves output_anim_seq in the DB
    ## - make the save

    # import pdb;pdb.set_trace()
    user_input = input("You: ")
    
    print("Chatbot:", user_input)
    prompt = user_input

    prompt_sequence.append("# " + prompt + "\n")
    error_prompt_sequence = prompt_sequence

    c, r, context = query_model(prompt_sequence, error_prompt_sequence, 0, context)
    # import pdb;pdb.set_trace()
    prompt_sequence.append(c)
    print(c)
    
    return context
    


if __name__ == "__main__":

    prompt_sequence = []
    context = {}
    
    hparams = IEFA_args()
    device = torch.device("cuda:" + str(hparams.device - 1)) if hparams.device > 0 else torch.device("cpu")
    
    print("---------------- Running IEFA editing pipeline ------------------")
    model_path = f"{os.path.join(hparams.root_dir, hparams.save_dir)}/{hparams.model_num}.pth"
    audio_path = "/input/inyup/IEFA/data/test/audio/userstudy/m03-angry-level_3-001.wav" # TODO : input in the exact audio file path you want to intput
    runner = Runner(hparams)
    
    runner.autoencoder.load_state_dict(torch.load(model_path, map_location='cuda:0')['autoencoder_state_dict'])
    runner.autoencoder.eval()
    
    ## loading direct neutral vtx animation sequence from captured data. Later will be replaced /w CodeTalker's prediction output
    f = open(os.path.join(hparams.con_data_root_dir, hparams.vtx_dtw_path), 'rb')
    vtx = pickle.load(f)
    # import pdb;pdb.set_trace()
    con_vtx_anim_seq = vtx["0"]["neutral_3_M003_front_neutral_3_001"]
    
    motion_id = 0
    
    prompt_sequence = read_progprompt_0(prompt_sequence) # this reads 'FacialMotion' class
    prompt_sequence = read_progprompt("",prompt_sequence)
    prompt_sequence = get_incontext(prompt_sequence)
    
    while True:
        context = run_pipeline(prompt_sequence, context) # 1. queries GPT / 2. run GPT created code 
        motion_info = context.get('db').load_motion(f"motion_{motion_id}", return_dict=True) # to keep track of exec() execution's state of Motion_DB()
        exp_bshp_anim_seq = motion_info["output_animation_seq"]
        exp_vtx_anim_seq = bshp_2_vtx(exp_bshp_anim_seq)
        render = True # switch on and off
        pred_vtx = direct_decoding(hparams=hparams,
                        motion_num = motion_id, 
                        con_vtx_anim_seq=con_vtx_anim_seq, 
                        exp_vtx_anim_seq=exp_vtx_anim_seq,
                        audio_path=audio_path, # for mux
                        runner=runner, 
                        render=render)
        motion_id += 1
        
"""
The person is talking. Make the face when you are bullied throughout the entire sequence. 
The person is talking. Make angry face throughout the entire sequence. Qickly blink in the middle. 
"""