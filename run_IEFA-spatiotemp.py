import torch
import os
import pickle
import numpy as np
from datetime import datetime
import sys
from argparse import ArgumentParser
import argparse
from src.FaceMEO.llm.motion_io import Motion_DB
from src.FaceMEO.llm.motion import FacialMotion
sys.path.append('./livelink_MEAD/') # added to access face_model_io in data_generation.py file
from data.livelink_MEAD import face_model_io
from src.FaceMEO.openai_wrapper import read_progprompt_0, read_progprompt, get_incontext, query_model
from src.disentanglement.train import Runner
from src.disentanglement.test import direct_decoding
from parser_util import IEFA_args
from data.livelink_MEAD.util import bshp_2_vtx

db = Motion_DB()

def run_pipeline(prompt_sequence, context, autorun=False, user_input : str = None):
    
    if not autorun:
        user_input = input("\nYou: ")
    
    prompt = user_input

    prompt_sequence.append("# " + prompt + "\n")
    error_prompt_sequence = prompt_sequence

    c, r, context = query_model(prompt_sequence, error_prompt_sequence, 0, context)
    # import pdb;pdb.set_trace()
    prompt_sequence.append(c)
    print(f"Chatbot: {c} \n")
    
    return context
    


if __name__ == "__main__":
    
    hparams = IEFA_args()
    device = torch.device("cuda:" + str(hparams.device - 1)) if hparams.device > 0 else torch.device("cpu")
    
    ## loading runner for autoencoder
    print("---------------- Running IEFA editing pipeline ------------------")
    model_path = f"{os.path.join(hparams.root_dir, hparams.save_dir)}/{hparams.model_num}.pth"
    
    ## for cpu only workspace, comment out
    runner = Runner(hparams)
    runner.autoencoder.load_state_dict(torch.load(model_path, map_location='cuda:0')['autoencoder_state_dict'])
    runner.autoencoder.eval()
    
    f = open(os.path.join(hparams.con_data_root_dir, hparams.vtx_dtw_path), 'rb') # ict_dataset_m003_vtx_dtw_nolevel_03
    vtx = pickle.load(f)
    # import pdb;pdb.set_trace()
    con_vtx_anim_seq_list = []
    len_list = []
    con_vtx_anim_seq = vtx["12"]["neutral_3_M003_front_neutral_3_003"]    # 45 (o)
    con_vtx_anim_seq_list.append(con_vtx_anim_seq)
    len_list.append(45)
    
    con_vtx_anim_seq = vtx["11"]["neutral_3_M003_front_neutral_3_040"]    # 126 (o)
    con_vtx_anim_seq_list.append(con_vtx_anim_seq)
    len_list.append(126)

    con_vtx_anim_seq = vtx["10"]["neutral_3_M003_front_neutral_3_039"]    # 99 (o)
    con_vtx_anim_seq_list.append(con_vtx_anim_seq)
    len_list.append(99)

    con_vtx_anim_seq = vtx["9"]["neutral_3_M003_front_neutral_3_038"]     # 150 (o) 
    con_vtx_anim_seq_list.append(con_vtx_anim_seq)
    len_list.append(150)

    con_vtx_anim_seq = vtx["8"]["neutral_3_M003_front_neutral_3_037"]     # 96 (o)
    con_vtx_anim_seq_list.append(con_vtx_anim_seq)
    len_list.append(96)

    con_vtx_anim_seq = vtx["7"]["neutral_3_M003_front_neutral_3_036"]     # 150 (o)
    con_vtx_anim_seq_list.append(con_vtx_anim_seq)
    len_list.append(150)

    con_vtx_anim_seq = vtx["6"]["neutral_3_M003_front_neutral_3_035"]     # 102 (o)
    con_vtx_anim_seq_list.append(con_vtx_anim_seq)
    len_list.append(102)

    con_vtx_anim_seq = vtx["5"]["neutral_3_M003_front_neutral_3_034"]     # 83 (o)
    con_vtx_anim_seq_list.append(con_vtx_anim_seq)
    len_list.append(83)

    con_vtx_anim_seq = vtx["4"]["neutral_3_M003_front_neutral_3_033"]     # 108 (o)
    con_vtx_anim_seq_list.append(con_vtx_anim_seq)
    len_list.append(108)

    con_vtx_anim_seq = vtx["3"]["neutral_3_M003_front_neutral_3_032"]     # 71 (o)
    con_vtx_anim_seq_list.append(con_vtx_anim_seq)
    len_list.append(71)

    con_vtx_anim_seq = vtx["2"]["neutral_3_M003_front_neutral_3_031"]     # 98 (o)
    con_vtx_anim_seq_list.append(con_vtx_anim_seq)
    len_list.append(98)

    con_vtx_anim_seq = vtx["1"]["neutral_3_M003_front_neutral_3_002"]     # 75 (o) 
    con_vtx_anim_seq_list.append(con_vtx_anim_seq)
    len_list.append(75)

    con_vtx_anim_seq = vtx["0"]["neutral_3_M003_front_neutral_3_001"]     # 111 (o) 
    con_vtx_anim_seq_list.append(con_vtx_anim_seq)
    len_list.append(111)
    
    audio_path_list = []
    audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-003.wav" # this was the correct audio
    audio_path_list.append(audio_path)
    
    audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-040.wav" # this was the correct audio
    audio_path_list.append(audio_path)

    audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-039.wav" # this was the correct audio
    audio_path_list.append(audio_path)

    audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-038.wav" # this was the correct audio
    audio_path_list.append(audio_path)

    audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-037.wav" # this was the correct audio
    audio_path_list.append(audio_path)

    audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-036.wav" # this was the correct audio
    audio_path_list.append(audio_path)

    audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-035.wav" # this was the correct audio
    audio_path_list.append(audio_path)

    audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-034.wav" # this was the correct audio
    audio_path_list.append(audio_path)

    audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-033.wav" # this was the correct audio
    audio_path_list.append(audio_path)

    audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-032.wav" # this was the correct audio
    audio_path_list.append(audio_path)

    audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-031.wav" # this was the correct audio
    audio_path_list.append(audio_path)

    audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-002.wav" # this was the correct audio
    audio_path_list.append(audio_path)

    audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-001.wav" # this was the correct audio
    audio_path_list.append(audio_path)
    
    ## args needed for loop
    prompt_sequence = []
    context = {}
    face_model = face_model_io.load_face_model('/source/inyup/ICT-FaceKit/FaceXModel') # switched to util_fast 
    # face_model=None
    final_mux_result_path = ""
    #################
    render = True # switch on and off
    render_only_result = True
    ####################
    motion_id = 1
    ## add preprompts
    prompt_sequence = get_incontext(prompt_sequence) # this reads context examples
    save_bshp_anim = False
    import time
    os.makedirs(f"./spatio_v1", exist_ok=True)
    for i, con_vtx_anim_seq in enumerate(con_vtx_anim_seq_list):
        # if i == 0:
        #     continue
        # import pdb;pdb.set_trace()
        ## edit texts
        audio_path = audio_path_list[i] # this was the correct audio
        #################
        ## temporal edits
        # neutral edit
        # edit1 = f"Add neutral expression at 0. Total length of frame is {len_list[i]}"
        # edit1 = f"Add neutral expression at 0. Total length of frame is {len_list[i]}"
        # edit1 = f"Add neutral expression at 0 and add sad expression at frame {len_list[i]-1} with intensity 2.0. Total length of frame is {len_list[i]}"
        # edit2 = f"Now clear all animation and just back to initialization. Then add neutral expression at frame 0 and sad expression at frame {len_list[i] / 2} with intensity 2.0." 
        #################
        ## spatial edits
        ## 0.2 is too small 
        # edit1 = f"slightly disgusted all along about intensity of 0.2. Just insert it at frame 1. Total length of frame is {len_list[i]}"
        # edit2 = f"Now clear all animation and just back to initialization. Now insert disgusted at frame 1 about intensity of 3.0. Total length of frame is {len_list[i]}. " 
        
        ## pair 1
        # edit1 = f"Initialize new motion. Total length of frame is {len_list[i]}. Slightly disgusted all along about intensity of 0.5. Just insert it at frame 1."
        # edit2 = f"Now insert disgusted at frame 1 again, but about intensity of 3.0. Total length of frame is {len_list[i]}. " 
        ## pair 2
        # edit1 = f"Initialize new motion. Total length of frame is {len_list[i]}. Lip_corner_puller about intensity of 0.5. Just insert it at frame 1."
        # edit2 = f"Now insert that face at frame 1 again, but about intensity of 2.0. Total length of frame is {len_list[i]}. " 
        ## pair 3
        edit1 = f"Initialize new motion. Total length of frame is {len_list[i]}. nose wrinkler about intensity of 0.5. Just insert it at frame 1."
        edit2 = f"Now insert that face at frame 1 again, but about intensity of 2.0. Total length of frame is {len_list[i]}. " 
        
        edits = []
        edits.append(edit1)
        edits.append(edit2)
        # motion_id = 1
        
        for j, edit in enumerate(edits):
            context = run_pipeline(prompt_sequence, context, autorun=True, user_input=edit) # 1. queries GPT / 2. run GPT created code 
            # import pdb;pdb.set_trace()
            ## new
            motion_key = f"motion_{motion_id}"
            motion_info = context.get('db').load_motion(motion_key, return_dict=True) # to keep track of exec() execution's state of Motion_DB()
            motion = FacialMotion(motion_info, len_list[i])
            exp_bshp_anim_seq = motion.output_animation_seq
            if save_bshp_anim:
                filename = datetime.now().date().strftime("%Y%m%d") + "_" + str(motion_id)
                np.save(f"/source/inyup/IEFA/data/test/result/vtx/{filename}_bshp_anim.npy", exp_bshp_anim_seq)
            exp_vtx_anim_seq = bshp_2_vtx(exp_bshp_anim_seq, face_model)
            # np.save(f"./spatio_v1/s{i}_e{j}_expression_vtx_seqence.npy", exp_vtx_anim_seq)
            if render:
                if render_only_result:
                    final_mux_result_path, pred_vtx = direct_decoding(hparams=hparams,
                                    motion_num = motion_id, 
                                    con_vtx_anim_seq=con_vtx_anim_seq, 
                                    exp_vtx_anim_seq=exp_vtx_anim_seq,
                                    src_vid_path = final_mux_result_path,
                                    face_model=face_model,
                                    audio_path=audio_path, # for mux
                                    runner=runner, 
                                    render=render,
                                    render_only_result=render_only_result,
                                    sentence_number=i,
                                    edit_number=j) 
                else:
                    final_mux_result_path, pred_vtx = direct_decoding(hparams=hparams,
                                    motion_num = motion_id, 
                                    con_vtx_anim_seq=con_vtx_anim_seq, 
                                    exp_vtx_anim_seq=exp_vtx_anim_seq,
                                    src_vid_path = final_mux_result_path,
                                    face_model=face_model,
                                    audio_path=audio_path, # for mux
                                    runner=runner, 
                                    render=render,
                                    render_only_result=render_only_result,
                                    sentence_number=i,
                                    edit_number=j)              
            else:
                pred_vtx = direct_decoding(hparams=hparams, 
                                motion_num = motion_id, 
                                con_vtx_anim_seq=con_vtx_anim_seq, 
                                exp_vtx_anim_seq=exp_vtx_anim_seq, 
                                src_vid_path = final_mux_result_path, 
                                face_model=face_model, 
                                audio_path=audio_path, # for mux 
                                runner=runner, 
                                render=render,
                                sentence_number=i,
                                edit_number=j) 
            motion_id += 1
