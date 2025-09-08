import torch
import os
import pickle
import numpy as np
from datetime import datetime
import sys
from src.FaceMEO.llm.motion_io import Motion_DB
sys.path.append('./livelink_MEAD/') # added to access face_model_io in data_generation.py file
from data.livelink_MEAD import face_model_io
from src.FaceMEO.openai_wrapper import read_progprompt_0, read_progprompt, get_incontext, query_model
from src.disentanglement.train import Runner
from src.disentanglement.test import direct_decoding, direct_decoding_o3d
from parser_util import IEFA_args

from data.livelink_MEAD.util import bshp_2_vtx

db = Motion_DB()

def run_pipeline(prompt_sequence, context):
            
    ## TODO 
    ## - (DONE) turned this into run each found methods one by one
    ## - check if this actually saves output_anim_seq in the DB
    ## - make the save

    # import pdb;pdb.set_trace()
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
    print("---------------- Running FaceDirector editing pipeline ------------------")
    model_path = f"{os.path.join(hparams.root_dir, hparams.save_dir)}/{hparams.model_num}.pth"
 
    runner = Runner(hparams)
    runner.autoencoder.load_state_dict(torch.load(model_path, map_location='cuda:0')['autoencoder_state_dict'])
    runner.autoencoder.eval()
    
    ## loading direct neutral vtx animation sequence from captured data. 
        # Later will be replaced /w CodeTalker's prediction output
        ###############
        # for my result
    f = open(os.path.join(hparams.con_data_root_dir, hparams.vtx_dtw_path), 'rb')
    vtx = pickle.load(f)
    con_vtx_anim_seq = vtx["0"]["neutral_3_M003_front_neutral_3_001"] 
    # audio_path = "/input/inyup/IEFA/data/test/audio/userstudy/m03-angry-level_3-001.wav" # input in the exact audio file path you want to mux
    audio_path = "/input/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-001.wav" # this was the correct audio
    
        ################
        # for comparison
    # f = open(os.path.join(hparams.con_data_root_dir, "lafa_ict_dataset_m003_vtx_v3.pickle"),'rb') 
    # f = open(os.path.join(hparams.con_data_root_dir, "lafa_dataset_m003_vtx_dtw_nolevel_v3.pickle"),'rb') 
    # f = open(os.path.join(hparams.con_data_root_dir, "m03-all-capture.pickle"),'rb') 
    # vtx = pickle.load(f)
    # import pdb;pdb.set_trace()

    # con_vtx_anim_seq = vtx["M003_front_angry_3_003"] # for comparison (test audio is angry_3_003 for comparison)
    # con_vtx_anim_seq = vtx["M003_front_angry_3_003"] 
    # f = open(os.path.join(hparams.con_data_root_dir, "m03-all-capture.pickle"),'rb') 
    # con_vtx_anim_seq = vtx["m03-neutral-level_1-003"]  # for my-all-capture.pickle
    # audio_path = "/input/inyup/IEFA/data/test/audio/userstudy/m03-angry-level_3-003.wav" # input in the exact audio file path you want to mux 

    ## args needed for loop
    prompt_sequence = []
    context = {}
    face_model = face_model_io.load_face_model('/input/inyup/ICT-FaceKit/FaceXModel') # switched to util_fast 
    # face_model=None
    final_mux_result_path = ""
    render = True # switch on and off
    motion_id = 1
    
    ## add preprompts
    prompt_sequence = read_progprompt_0(prompt_sequence) # this reads 'FacialMotion' class
    prompt_sequence = read_progprompt("",prompt_sequence) # this reads independant examples
    prompt_sequence = get_incontext(prompt_sequence) # this reads context examples
    save_bshp_anim = False
    import time
    while True:
        
        context = run_pipeline(prompt_sequence, context) # 1. queries GPT / 2. run GPT created code 
        motion_info = context.get('db').load_motion(f"motion_{motion_id}", return_dict=True) # to keep track of exec() execution's state of Motion_DB()
        exp_bshp_anim_seq = motion_info["output_animation_seq"]
        if save_bshp_anim:
            filename = datetime.now().date().strftime("%Y%m%d") + "_" + str(motion_id)
            np.save(f"/input/inyup/IEFA/data/test/result/vtx/{filename}_bshp_anim.npy", exp_bshp_anim_seq)
        exp_vtx_anim_seq = bshp_2_vtx(exp_bshp_anim_seq, face_model)
        if render:
            # final_mux_result_path, pred_vtx = direct_decoding(hparams=hparams,
            #                 motion_num = motion_id, 
            #                 con_vtx_anim_seq=con_vtx_anim_seq, 
            #                 exp_vtx_anim_seq=exp_vtx_anim_seq,
            #                 src_vid_path = final_mux_result_path,
            #                 face_model=face_model,
            #                 audio_path=audio_path, # for mux
            #                 runner=runner, 
            #                 render=render) 
            final_mux_result_path, pred_vtx = direct_decoding_o3d(
                
                
                
            )
        else:
            # pred_vtx = direct_decoding(hparams=hparams,
            #                 motion_num = motion_id, 
            #                 con_vtx_anim_seq=con_vtx_anim_seq, 
            #                 exp_vtx_anim_seq=exp_vtx_anim_seq,
            #                 src_vid_path = final_mux_result_path,
            #                 face_model=face_model,
            #                 audio_path=audio_path, # for mux
            #                 runner=runner, 
            #                 render=render) 
            final_mux_result_path, pred_vtx = direct_decoding_o3d(
                
                
                
            )
        motion_id += 1

