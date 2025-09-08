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
    print("---------------- Running IEFA editing pipeline ------------------")
    model_path = f"{os.path.join(hparams.root_dir, hparams.save_dir)}/{hparams.model_num}.pth"
 
    runner = Runner(hparams)
    runner.autoencoder.load_state_dict(torch.load(model_path, map_location='cuda:0')['autoencoder_state_dict'])
    runner.autoencoder.eval()
    
    ## loading direct neutral vtx animation sequence from captured data. 
        # Later will be replaced /w CodeTalker's prediction output
        ###############
        # for my result
        
    f = open(os.path.join(hparams.con_data_root_dir, hparams.vtx_dtw_path), 'rb')
    # f = open("/input/inyup/IEFA/data/feature/ict_dataset_m003_vtx_dtw_gf_nolevel_03.pickle", 'rb')
    vtx = pickle.load(f)
    # import pdb;pdb.set_trace()
    source_path = "/source/inyup/IEFA/data/livelink_MEAD"
    # con_vtx_anim_seq = vtx["12"]["neutral_3_M003_front_neutral_3_003"]    # 45 (o)
    con_vtx_anim_seq = vtx["11"]["neutral_3_M003_front_neutral_3_040"]    # 126 (o)
    # con_vtx_anim_seq = vtx["10"]["neutral_3_M003_front_neutral_3_039"]    # 99 (o)
    # con_vtx_anim_seq = vtx["9"]["neutral_3_M003_front_neutral_3_038"]     # 150 (o) 
    # con_vtx_anim_seq = vtx["8"]["neutral_3_M003_front_neutral_3_037"]     # 96 (o)
    # con_vtx_anim_seq = vtx["7"]["neutral_3_M003_front_neutral_3_036"]     # 150 (o)
    # con_vtx_anim_seq = vtx["6"]["neutral_3_M003_front_neutral_3_035"]     # 102 (o)
    # con_vtx_anim_seq = vtx["5"]["neutral_3_M003_front_neutral_3_034"]     # 83 (o)
    # con_vtx_anim_seq = vtx["4"]["neutral_3_M003_front_neutral_3_033"]     # 108 (o)
    # con_vtx_anim_seq = vtx["3"]["neutral_3_M003_front_neutral_3_032"]     # 71 (o)
    # con_vtx_anim_seq = vtx["2"]["neutral_3_M003_front_neutral_3_031"]     # 98 (o)
    # con_vtx_anim_seq = vtx["1"]["neutral_3_M003_front_neutral_3_002"]     # 75 (o) 
    # con_vtx_anim_seq = vtx["0"]["neutral_3_M003_front_neutral_3_001"]     # 111 (o) 
    
    
    ##########################
    ## for EMOTE comparison ##
    ##########################
    ## make very angry face at frame 10 for intenstiy 3.0
    ## make very contempt face at frame 10 for intenstiy 3.0
    ## make very disgusted face at frame 10 for intenstiy 3.0
    ## make very fearful face at frame 10 for intenstiy 3.0
    ## make very happy face at frame 10 for intenstiy 3.0
    ## make very sad face at frame 10 for intenstiy 3.0
    ## make very surprised face at frame 10 for intenstiy 3.0
    
    ###################################
    ## for naive additive comparison ##
    ###################################
    ## open mouth with intensity 1.0 at frame 10
    ## surprised face at frame 10

    # con_vtx_anim_seq = np.load(os.path.join(source_path,"12/neutral_1_003/trim_bshp_neutral_raw.npy"))   
    # con_vtx_anim_seq = np.load(os.path.join(source_path,"11/neutral_1_040/trim_bshp_neutral_raw.npy"))   
    # con_vtx_anim_seq = np.load(os.path.join(source_path,"10/neutral_1_039/trim_bshp_neutral_raw.npy"))   
    # con_vtx_anim_seq = np.load(os.path.join(source_path,"9/neutral_1_038/trim_bshp_neutral_raw.npy"))   
    # con_vtx_anim_seq = np.load(os.path.join(source_path,"8/neutral_1_037/trim_bshp_neutral_raw.npy"))   
    # con_vtx_anim_seq = np.load(os.path.join(source_path,"7/neutral_1_036/trim_bshp_neutral_raw.npy"))   
    # con_vtx_anim_seq = np.load(os.path.join(source_path,"5/neutral_1_034/trim_bshp_neutral_raw.npy"))   
    # con_vtx_anim_seq = np.load(os.path.join(source_path,"4/neutral_1_033/trim_bshp_neutral_raw.npy"))   
    # con_vtx_anim_seq = np.load(os.path.join(source_path,"3/neutral_1_032/trim_bshp_neutral_raw.npy"))   
    # con_vtx_anim_seq = np.load(os.path.join(source_path,"2/neutral_1_031/trim_bshp_neutral_raw.npy"))   
    # con_vtx_anim_seq = np.load(os.path.join(source_path,"1/neutral_1_002/trim_bshp_neutral_raw.npy"))   
    # con_vtx_anim_seq = np.load(os.path.join(source_path,"0/neutral_1_001/trim_bshp_neutral_raw.npy"))   

    # audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-003.wav" # this was the correct audio
    audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-040.wav" # this was the correct audio
    # audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-039.wav" # this was the correct audio
    # audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-038.wav" # this was the correct audio
    # audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-037.wav" # this was the correct audio
    # audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-036.wav" # this was the correct audio
    # audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-035.wav" # this was the correct audio
    # audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-034.wav" # this was the correct audio
    # audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-033.wav" # this was the correct audio
    # audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-032.wav" # this was the correct audio
    # audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-031.wav" # this was the correct audio
    # audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-002.wav" # this was the correct audio
    # audio_path = "/source/inyup/IEFA/data/test/audio/userstudy/m03-neutral-level_1-001.wav" # this was the correct audio
    
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
    face_model = face_model_io.load_face_model('/source/inyup/ICT-FaceKit/FaceXModel') # switched to util_fast 
    # face_model=None
    final_mux_result_path = ""
    #################
    render = True # switch on and off
    render_only_result = False
    ####################
    motion_id = 1
    
    ## add preprompts
    # prompt_sequence = read_progprompt_0(prompt_sequence) # this reads 'FacialMotion' class
    # prompt_sequence = read_progprompt("",prompt_sequence) # this reads independant examples
    prompt_sequence = get_incontext(prompt_sequence) # this reads context examples
    save_bshp_anim = False
    import time
    while True:
        
        context = run_pipeline(prompt_sequence, context) # 1. queries GPT / 2. run GPT created code 
        
        ## old 
        # motion_info = context.get('db').load_motion(f"motion_{motion_id}", return_dict=True) # to keep track of exec() execution's state of Motion_DB()
        # exp_bshp_anim_seq = motion_info["output_animation_seq"]
        import pdb;pdb.set_trace()
        ## new
        motion_key = f"motion_{motion_id}"
        motion_info = context.get('db').load_motion(motion_key, return_dict=True) # to keep track of exec() execution's state of Motion_DB()
        motion = FacialMotion(motion_info)
        exp_bshp_anim_seq = motion.output_animation_seq
        if save_bshp_anim:
            filename = datetime.now().date().strftime("%Y%m%d") + "_" + str(motion_id)
            np.save(f"/source/inyup/IEFA/data/test/result/vtx/{filename}_bshp_anim.npy", exp_bshp_anim_seq)
        exp_vtx_anim_seq = bshp_2_vtx(exp_bshp_anim_seq, face_model)
        # if render:
        #     if render_only_result:
        #         final_mux_result_path, pred_vtx = direct_decoding(hparams=hparams,
        #                         motion_num = motion_id, 
        #                         con_vtx_anim_seq=con_vtx_anim_seq, 
        #                         exp_vtx_anim_seq=exp_vtx_anim_seq,
        #                         src_vid_path = final_mux_result_path,
        #                         face_model=face_model,
        #                         audio_path=audio_path, # for mux
        #                         runner=runner, 
        #                         render=render,
        #                         render_only_result=render_only_result) 
        #     else:
        #         final_mux_result_path, pred_vtx = direct_decoding(hparams=hparams,
        #                         motion_num = motion_id, 
        #                         con_vtx_anim_seq=con_vtx_anim_seq, 
        #                         exp_vtx_anim_seq=exp_vtx_anim_seq,
        #                         src_vid_path = final_mux_result_path,
        #                         face_model=face_model,
        #                         audio_path=audio_path, # for mux
        #                         runner=runner, 
        #                         render=render,
        #                         render_only_result=render_only_result)              
        # else:
        #     pred_vtx = direct_decoding(hparams=hparams, 
        #                     motion_num = motion_id, 
        #                     con_vtx_anim_seq=con_vtx_anim_seq, 
        #                     exp_vtx_anim_seq=exp_vtx_anim_seq, 
        #                     src_vid_path = final_mux_result_path, 
        #                     face_model=face_model, 
        #                     audio_path=audio_path, # for mux 
        #                     runner=runner, 
        #                     render=render) 
        motion_id += 1

"""
The person is talking. Show fear near the start of the sequence. 
When mouth is opened largest, close both eyes for a second.
Open longer.

0911
The person is talking. Change to disgusted face all along. 
At almost the end, close eyes and maintain that way. 
Move that closing of eyes to the middle of the sequence. 
Close only left eye at the start for a second. 
Undo all edits.

0912
The person is talking. Smirk left till the end. 
Raise right eyebrows from the start about a second and a half.
Make a face when you are watching a two grown ups fighting for a cookie at almost the end.
Mouth is opened too big, close a little. 
Undo previous edit.
Make the looking at fighting for cookie face longer.
Raise the other brow as well.
Raise it at the same point when raising left brow
undo all brow edits

0916
The person is talking. Show face when a little kid found out Santa doesn't exist along the whole sequence.
Blink in the middle. 
Blink longer.
Add blink almost at front.
Close mouth in the middle and maintain that till the end. 

0919
The person is talking. Show face when someone saw a grizzly walking by the street during the whole sequence. 
Squint both eyes from the start to the middle. 

0923
The person is talking. Show contempt face in the middle for a second.
Close right eyes almost at the end.
Smirk right almost at the front for 0.5 second.
Add eye squint in the beginning till the middle.
Both lip corners down at the start for the same length with eye squint.
Raise left brow at front as well.
Lower the other one at the same point and the same length of time.
Make that previous contempt face longer till the end.

0924
The person is talking. Show contempt face in the middle for a second.
Close right eyes almost at the end.
Close the other eye at the same time and length.
Sneer you right nose almost at the front for 0.5 second.
Raise left brow with that nose sneer. 
Raise the other brow as well.
Move eye right at frame 35. 

The person is talking. Fearful in the middle.
Make the face 2.5 times longer.
three times longer. >> Doesn't work, right;; should specifiy frames 
Make it for a 2 second. 
make the face slower. change should take a second.

0925
The person is talking. Fearful in the middle.
Close right eyes almost at the end.
Close the other eye at the same time and length.
Sneer you right nose almost at the front for 0.5 second.
The other nose as well.

0926
The person is talking. Fearful in the middle.
Close right eyes almost at the end.
close the other one at the same point with the same length and speed.
Puff your right cheek at front for 0.8 sec.
The other cheek too.
Lower left brow at same point for a sec.
The other brow as well. 
Lower those two longer for a second.

1002
The person is talking. Lower your left brow down a lot for a second at front.
Look inward only left for 0.5 sec in the middle.
Look outward only right at almost the end for 25 frames.
Blink only left at frame 15.
Another left eye blink at the end for a second.
Mouth dimple left at frame 15 for a second.
Lower left mouth down at almost the end.
Make you upper mouth up for only left at frame 40 about a 0.5 sec.
Sneer you nose right at frame 40 for a second.

1020
The person is talking. You failed the job interview. Put it in the middle. 
Lower outerbrows more.
Undo
Fearful in the middle. 

1102 (succesfull) 
# reverse       - done
# all at once   - 
The person is talking. Fearful in the middle.
Smirk left at frame 10 for a second. 
Raise eyebrows at frame 20 for 0.5 second.
Smirk the other side as well,
cheek puff when the mouth is biggest.

1103 for dataset ablation
The person is talking. Blink in the middle for a second.
Blink only left at frame 10.
sneer nose in the middle for a second.
Make face when messi lost at world cup final at almost the end. 
bshp  (ict_vtx_bshpAct_facsAct_warmup_nofacs.pth)
bshp facs (ict_vtx_bshpAct_facsAct.pth)
ours (ict_vtx_warmup2_warmup_bshp_facs200_emot50_l1lip_ld10_loo01675.pth)

1104 for lip loss ablation
# no lip (ict_vtx_warmup2_warmup_bshp_facs200_emot50_nolip.pth)
# ours (BEST WIP)
The person is talking. Blink in the middle for a second.

0104 checking for lip loss performance
The person is talking. Open your jaw a little for all along the sequnece

0105 checking for per edit spatial edits
The person is talking. slightly Smile only at the mouth in the middel of the sequence. 

0113 User Study
## scene #1 (DDNE)
% 1. Scene 1: A Joyful Celebration
% Artist Intent: Capture a transition from calmness to a joyful, expressive celebration.
% Edit 1 (Adding * Temporal):
Start with a neutral expression, then transition to a sligh lip corner smile at almost the end of the sequence (frame 90).
% Semantic Point: Ensure the mouth reaches its widest stretch at the last frame.
% Edit 2 (Refine * Global):
Make that face at edit 1 more joyful overall by raising both cheeks (cheek raiser) and widen eyes more
% Edit 3 (Adding * Local):
At the midpoint of the sequence, raise the both inner and outer eyebrows (outer brow raiser) to enhance the cheerful tone.
% Edit 4 (Refine * Local):
Make the smile at edit 1 more intense to make both corners smile.

## scene #2 (DONE)
% 2. Scene 2: A Sudden Realization
% Artist Intent: Depict a character transitioning from surprise to focused determination.
% Edit 1 (Adding * Global):
% "Add a surprised expression in the first quarter of the sequence (frame 30), with wide eyes (eye wide) and slightly raised inner eyebrows (inner brow raiser)."
% Edit 2 (Adding * Temporal):
% "Make the surprised expression last until halfway through the sequence, then transition into a neutral expression at the end."
% Edit 3 (Refine * Temporal):
% "Make the surprise appear earlier and more quickly, ensuring the wide eyes and raised eyebrows peak at frame 20."
% Edit 4 (Refine * Local):
% "In the final frames, lower the outer eyebrows (brow lowerer) slightly to add a hint of determination."

## scene #3 (DDNE)

% Edit 1 (Adding * Temporal):
Begin with a neutral expression, then slowly build tension by narrowing the eyes (lid tightener) and lowering the eyebrows (brow lowerer) in the last third of the sequence.(frame 90)
% Edit 2 (Adding * Local):
At frame 60, add a lip press (lip presser) to enhance the sense of nervousness
% Edit 3 (Refine * Temporal):
Increase the speed of the tension buildup, making the narrowed eyes appear earlier, around frame 50.
% Edit 4 (Refine * Local):
Toward the end, lower the inner eyebrows (brow lowerer) more significantly to emphasize the nervousnes

## scene #4 (DONE)
% 4. Scene 4: A Bittersweet Goodbye
% Artist Intent: Transition from sadness to a slight smile, reflecting acceptance.
% Edit 1 (Adding * Global):
% "Start with a neutral expression and transition to a sad expression by the middle of the sequence, with a frown (mouth frown) and slightly lowered inner eyebrows (brow lowerer)."
% Edit 2 (Refine * Local):
% "At the peak of the sad expression (frame 45), make the inner brows rise slightly (inner brow raiser) to add a touch of vulnerability."
% Edit 3 (Adding * Temporal):
% "Toward the end (last quarter), make the happy face."
% Edit 4 (Refine * Global):
% "Blend the sad and smiling expressions seamlessly, ensuring the sadness lingers slightly while the smile emerges."

## scene #5 (DONE)
% 5. Scene 5: Mischievous Confidence
% Artist Intent: Create a playful expression, starting with confidence and ending with a smirk.
% Edit 1 (Adding * Global):
Add a confident expression with slightly raised outer eyebrows (outer brow raiser) and a upward curve of the right corner of the mouth slightly (about intensity 0.5) and mouth moved to right slightly as well at almost the begining of the sequence.
% Edit 2 (Refine * Local):
% Deepen the right smile and the moving of mouth right more.
% Edit 3 (Adding * Local):
At frame 90, add a wink on the right eye (eye blink), meaning keyframe duration 2. 
% Edit 4 (Refine * Temporal):
Make the face go longer for about a second. Meaning the duration should be 30 frame. 

## scene #6 (   )
% 6. Scene 6: Harry’s First Spell
% Artist Intent: Capture Harry's emotions when he successfully casts his first spell, transitioning from surprise to pride and joy.
% Edit 1 (Adding * Global):
% "Make the character look like Harry Potter just cast his first spell, with a wide-eyed surprised expression and slightly parted lips."
% Semantic Point: Ensure the eyes reach their maximum wideness at frame 30.
% Edit 2 (Adding * Temporal):
% "Hold the surprised expression until the midpoint of the sequence, then transition into a proud smile toward the end."
% Timing: Surprise fades at frame 45, pride peaks at frame 90.
% Edit 3 (Refine * Global):
% "At the proudest moment (frame 75), raise the outer eyebrows slightly and enhance the smile with higher mouth corners."
% Edit 4 (Refine * Local):
% "In the final frames, slightly adjust the cheeks to lift more (cheek raiser), emphasizing the joy."

## scene #7 (DONE)
Artist Intent: Capture the moment when anger builds up quickly, leading to an intense expression of frustration.
Edit 1 (Adding * Global):
"Introduce a neutral face at the beginning, then, within the first third of the sequence, build up to anger with furrowed brows (brow lowerer), tightened eyelids (lid tightener), and a tense slight jaw (jaw thrust) about 0.6 intensity."
Edit 2 (Adding * Local):
"At frame 60, add a slight left nose wrinkle and right nose wrinkle for intensity 1.0 to intensify the frustration."
Edit 3 (Refine * Temporal):
"The winkle disappears too fast. Make it last for a second. Meaning the keyframe length should be 30 frames."
Edit 4 (Refine * Local):
"Toward the end, and add a cheek puff to reinforce the feeling of suppressed rage."

## scene #8 (WIP)
% Artist Intent: Show a character reacting to an unexpected and magical event, shifting from confusion to excitement.
% Edit 1 (Adding * Global):
% "Make the character react as if they just saw a dragon flying past, with a confused expression: furrowed brows (brow lowerer) and slightly open mouth."
% Semantic Point: The mouth should be at its maximum openness (jaw open) by frame 20.
% Edit 2 (Adding * Local):
% "At the middle of the sequence, add a subtle upward flick of the left eyebrow (outer brow raiser) to convey curiosity."
% Edit 3 (Refine * Temporal):
% "Make the expression of confusion transition faster into excitement, with the excitement peaking at frame 60."
% Edit 4 (Refine * Global):
% "Enhance the excitement by adding a big smile (mouth smile) while slightly raising both cheeks (cheek raiser) toward the end."

## scene #9 (DONE)
Scene #9: The Hidden Sorrow
Artist Intent: Express a deep sorrow that is not immediately visible—outwardly composed but carrying a heavy emotional weight inside.
Edit 1 (Adding * Global):
Start with a neutral expression and gradually transition into a subtle sadness in the middle of the sequence. Lower the mouth corners (lip corner depressor) and furrow the brows (brow lowerer) to convey inner grief.
Edit 2 (Refine * Local):
At the midpoint (frame 50), enhance the emotional depth by strongly tightening the eyelids for about intensity of 1.5 (eye squint) as if holding back tears.
Edit 3 (Adding * Temporal)
Let the sadness linger toward the end but fade slightly, making the mouth relax just before the last frame to suggest quiet acceptance. 
Edit 4 (Refine * Global):
Softly raise the inner eyebrows (inner brow raiser) and kiss (mouth pucker) at the last few frames(frame 100) to add a touch of vulnerability, but with love. (지금 이거 )


## 0118
for ours-iter figure
Add a left wink and a left smirk at frame 10. Make that wink last 1 frame meaning the duration should be 1.
"""
