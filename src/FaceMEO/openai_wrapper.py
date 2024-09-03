
from openai import OpenAI
import os 
import time
import ast
import json
import numpy as np
# import pose_mask
import torch
import sys
import traceback
# from llm.assertions import *
if __name__ != "main":
    sys.path.append("/input/inyup/IEFA/src/FaceMEO/")
from llm.motion import FacialMotion
from llm.motion_io import Motion_DB

client = OpenAI(
    api_key="" # VML public key
)

messages = []
def query_gpt():
    MODEL = "gpt-4o" 
    # MODEL = "gpt-3.5-turbo"
    response = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    temperature=0)
    content = response.choices[0].message.content
    return content

def sequence_content(content, prompt2, append=True):
    if(append):
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": prompt2})
    else:
        messages[-2] = {"role": "assistant", "content": content}
        messages[-1] = {"role": "user", "content": prompt2}

def query_gpt_sequential(prompt_sequence):
        responses = []
        for i in range(0, len(prompt_sequence) - 1, 2):
            messages.append({"role": "user", "content": prompt_sequence[i]})
            messages.append({"role": "assistant", "content": prompt_sequence[i + 1]})
        messages.append({"role": "user", "content": prompt_sequence[-1]})
        content = query_gpt()
        responses.append(content)
        messages.clear()
        return responses[-1]

def response_to_code(responses, err_prompt_sequence, try_counter, logger = None, trylist=[], context = {}):
    """
    This function processes the responses from the GPT model, 
    attempts to execute code snippets found in the responses, 
    and handles errors if they occur. 
    It also manages retries up to a certain limit (try_counter)
    """
    if try_counter >= 3:
        print("Giving up")
        return None, -1
    responses_split = responses.split("\n")
    methods = []
    valid = True
    not_defined = False
    found_load_motion = False
    found_save_motion = False
    print(responses)
    counter = 0

    ## TODO ##
    ## - (DONE) change this to retrieve face version
    for response in responses_split:
        # if "do_" in response and "(" in response and "undo" not in response:
        #     methods.append(response.strip())
        if "set_" in response or "activate_" in response or "FacialMotion(" in response:
            methods.append(response.strip())
        elif " = [" in response or "= \" " in response:
            methods.append(response.strip())
        elif " = " in response:
            methods.append(response.strip())
        elif "load_motion" in response:
            methods.append(response.strip())
            found_load_motion = True
        elif "save_motion" in response:
            methods.append(response.strip())
            found_save_motion = True
            
    ## this is the part where created code is actually executed for test -> assertions need to be set(completed)
    for method in methods:
        # import pdb;pdb.set_trace()
        try:
            print(method)
            # success, err = eval(method) # attempts to execute the method using Python's eval() function, which evaluates a string as Python code
            exec(method, globals(), context) # assignment operation only works with 'exec' and globals() needed to let exec() know current imported modules, and context for retrieving exec()'s retrived global variables
            # if success < 0: # means it's error
            #     err_prompt_sequence.append(responses)
            #     tb = traceback.format_exc() # logs the traceback (tb)
            #     err_prompt_sequence.append(err)
            #     valid = False
            #     break
        except Exception as err: # catches any exceptions raised during the execution of eval()
            print("try except", err) 
            err_prompt_sequence.append(responses)
            tb = traceback.format_exc()
            err_prompt_sequence.append(str(err))
            valid = False
            break
    # import pdb;pdb.set_trace()
    ## checks any methods were found or if both load_motion and save_motion were detected
    found_methods = False
    # import pdb;pdb.set_trace()
    if len(methods) > 0: 
        found_methods = True
    elif len(methods) == 0 and (found_load_motion or found_save_motion):
        found_methods = True
    else:
        print(methods) # likely empty, b/c no methods

    if not found_methods and not not_defined :
        print("Invalid Program")
        print(responses)
        err_prompt_sequence.append(responses)
        err_prompt_sequence.append("Please respond by editing your invalid program.") # asking the GPT model to edit the invalid program.
        valid = False
    trylist.append(counter) # trylist keeps track of the attempts
    if not valid: # If the response was not valid
        # import pdb;pdb.set_trace()
        # if logger:
        #     logger.log_error(responses)
        code, responses, context = query_model(err_prompt_sequence, err_prompt_sequence, try_counter + 1, logger, trylist, context) # to retry with an updated err_prompt_sequence, incrementing try_counter by 1.
        counter += 1
        #trylist.append(counter)
    else: # # If the response was valid, it assigns the responses to code.
        code = responses
    # import pdb;pdb.set_trace()

    return code, counter, context

prompt_sequence = []
def query_model(prompt, err_prompt_sequence, try_counter, logger = None, trylist=[], context = {}):
    print("querying model")
    responses = query_gpt_sequential(prompt)
    code , responses, context = response_to_code(responses, err_prompt_sequence, try_counter, logger, trylist, context)
    return code, responses, context

def read_progprompt(edit_instruction, prompt_sequence):
    print(edit_instruction)
    if __name__ == "__main__":
        path = "llm/prog_prompt3.py"
    else: # case when run from run_IEFA.py
        path = "src/FaceMEO/llm/prog_prompt3.py"
    with open(path, "r") as f:
        lines = f.read()
        prompt_sequence.append("```python\n" + lines + "```")
    return prompt_sequence

def read_progprompt_0(prompt_sequence):
    if __name__ == "__main__":
        path = "llm/motion.py"
    else: # case when run from run_IEFA.py
        path = "src/FaceMEO/llm/motion.py"
    with open(path, "r") as f:
            lines = f.read()
            prompt_sequence.append("```python\n" + lines + "```")
    return prompt_sequence

def get_incontext(prompt_sequence):
    prompt_sequence[-1] += "# the person is talking. Make a face like when Harry Potter is fighting with Voldmort right throughout the entire motion.\n"
    prompt_sequence.append('''
        # initialize motion DB
        db = Motion_DB()
        def angry_disgusted():
            # initialize facial animation with speech 
            motion_1 = FacialMotion()
            
            # the original motion is that the person talking. 
            # the desired edit is to ativate Harry Potter is fighting with Voldmort face in the middle of the motion
            
            # the involved identifiers are 'angry' and 'disgusted'
            # no specification on intensity, so default
            identifiers = ["angry", "disgusted"]
            motion_1.activate_hierarchical(identifiers=identifiers)
            # 'throughout the entire motion' indicates 0 in 'insert_point', so default
            # 'throughout the entire motion' also indicates sequence length for keyframe speed, so seq_len
            speed_locations = ["keyframes"]
            speed = [motion_1.seq_len]
            motion_1.set_speed(speed_locations, speed, replace=True) # replace, not multiplication
            
            motion_1.activate_blendshape()
        
            # save motion
            db.save_motion(motion_1, "motion_1")
    ''')

    prompt_sequence.append("# Make the face look angrier.\n")
    prompt_sequence.append('''def angrier():
        # load motion
        motion_1 = db.load_motion("motion_1")
        motion_2 = FacialMotion(animation_seq=motion_1.output_animation_seq)

        # the original motion was that the person is talking and was edited to have a face like when Harry Potter is fighting with Voldmort right from the start. 
        # the desired edit is to make the keyframe face look angrier
        
        # no extra identifiers detected, reusing parameters... 
        motion_2.set_parameter(motion_1)
        # 'angrier' indicates higher in 'intensity', so x1.5 in intensity
        intensities = [1.5]
        motion_2.activate_hierarchical(motion_1.cur_activated, intensities)
        
        motion_2.activate_blendshape()

        # save motion 
        db.save_motion(motion_2, "motion_2")
    '''
    )
    prompt_sequence.append("# Lower brows when jaw is max opened.\n")
    prompt_sequence.append('''def brow_lower_max_jaw():
        # load motion
        motion_2 = db.load_motion("motion_2")
        motion_3 = FacialMotion(animation_seq=motion_2.output_animation_seq)

        # he original motion was that the person is talking and was edited to have a face like when Harry Potter is fighting with Voldmort right from the start and the face got angrier. 
        # the desired edit is to lower brwos when jaw is max opened.
        
        # identifiers invovlved are 'brow_lowerer'
        # no specification on intensity, so default
        identifiers = ["brow_lowerer"]
        motion_3.activate_hierarchical(identifiers=identifiers)
        # no specification on speed_location, so default
        # 'when jaw is max opened' indicates 'max' value for 'jaw_open'
        point = "jaw_open" 
        motion_3.set_insert_point(motion_2.when_bshp("max", point))


        motion_3.activate_blendshape()

        # save motion 
        db.save_motion(motion_3, "motion_3")
    '''
    )

    prompt_sequence.append("# Lower the right brows more.\n")
    prompt_sequence.append('''def right_brow_lower():
        # load motion
        motion_3 = db.load_motion("motion_3")
        motion_4 = FacialMotion(animation_seq=motion_3.output_animation_seq)
        
        # the original motion was that the person was talking and was edited to have a face like when Harry Potter is fighting with Voldmort right from the start and make it angrier and lower both brows in the middle.  
        # The desired edit is to lower the right brow more
        
        # involved identifiers are "brow_down_r" 
        identifiers = ["brow_down_r"]
        # lower more means high in intensity, x1.5 for the current 'brow_down_r" related value 
        intensities = [1.5]
        motion_4.activate_hierarchical(identifiers, intensities) 
        # no specification on speed_location, so default
        # no specification on insert_point, so default
        
        motion_4.activate_blendshape()
        
        # save motion 
        db.save_motion(motion_4, "motion_4")
    '''
    )
    
    prompt_sequence.append("# Other brows too, at the same time.\n")
    prompt_sequence.append('''def other_brow_lower():
        # load motion
        motion_4 = db.load_motion("motion_4")
        motion_5 = FacialMotion(animation_seq=motion_4.output_animation_seq)
        
        # the original motion was that the person was talking and was edited to have a face like when Harry Potter is fighting with Voldmort right from the start and make it angrier and lower both brows in the middle and lower right brows more. 
        # The desired edit is to lower the other brow more as well at the same time
        
        # involved identifiers are "brow_down_r" 
        identifiers = ["brow_down_r"]
        # lower more means high in intensity, x1.5 for the current 'brow_down_r" related value 
        intensities = [1.5]
        motion_5.activate_hierarchical(identifiers, intensities) 
        # no specification on speed_location, so default
        # no specification on insert_point, so default
        
        motion_5.activate_blendshape()
        
        # save motion 
        db.save_motion(motion_5, "motion_5")
    '''
    )

    prompt_sequence.append("# smirk on the left side when right brow is at lowest\n")
    prompt_sequence.append('''def smirk_left():
        # load motion
        motion_5 = db.load_motion("motion_5")
        motion_6 = FacialMotion(animation_seq=motion_5.output_animation_seq)
        
        # the original motion was that the person was talking and was edited to have a face like when Harry Potter is fighting with Voldemort right from the start, made it angrier, lowered both brows in the middle, lowered the right brow more. And Other brows too, at the same time.
        # The desired edit is to add an asymmetric smirk on the left side of the face when right brow is at lowest.
        
        # involved identifiers are "mouth_smile_l"
        # no specification on intensity, so default
        identifiers = ["mouth_smile_l"]
        motion_6.activate_hierarchical(identifiers=identifiers)
        # no specification on speed_location, so default 
        # 'when right brow is at lowest' indicates 'max' value for 'brow_down_r'
        point = "brow_down_r" 
        motion_6.set_insert_point(motion_6.when_bshp("max", point)) 
        
        motion_6.activate_blendshape()
        
        # save motion
        db.save_motion(motion_6, "motion_6")
    '''
    )
    
    prompt_sequence.append("# smirk slower\n")
    prompt_sequence.append('''def smirk_left():
        # load motion
        motion_6 = db.load_motion("motion_6")
        motion_7 = FacialMotion(animation_seq=motion_6.output_animation_seq)
        
        # the original motion was that the person was talking and was edited to have a face like when Harry Potter is fighting with Voldemort right from the start, made it angrier, lowered both brows in the middle, lowered the right brow more. And Other brows too, at the same time. Smirk on the left side when right brow is at lowest 
        # The desired edit is to add an asymmetric smirk on the left side of the face when right brow is at lowest.
        
        # no extra identifiers detected, reusing parameters... 
        motion_7.set_parameter(motion_6)
        # 'slower' indicates longer number of frames in 'front_speed' frames in 'speed', so x1.5 the default
        speed_locations = ["front_speed"]
        speed = [1.5]
        motion_7.set_speed(speed_locations, speed)
        
        motion_7.activate_blendshape()
        
        # save motion
        db.save_motion(motion_7, "motion_7")
    '''
    )
    
    prompt_sequence.append("# Undo all brow edits\n")
    prompt_sequence.append('''def undo_brows():       
        # reverting the edit to before the editing of brows. Do it by loading previous motion, motion_2.
        motion_2 = db.load_motion("motion_2")
        motion_8 = FacialMotion(animation_seq=motion_2.output_animation_seq)
        motion_8.set_parameter(motion_2)
        
        # save it without any changes 
        db.save_motion(motion_8, "motion_8")
    '''
    )
    
    prompt_sequence.append("# Do angry face starting from the middle, not from the start.\n")
    prompt_sequence.append('''def angry_start_middle():

        motion_8 = db.load_motion("motion_8")
        motion_9 = FacialMotion(animation_seq=motion_8.output_animation_seq)

        # the original motion was that the person was talking and was edited to have a face like when Harry Potter is fighting with Voldemort right from the start, made it angrier.
        # the primary joints in volved are the left shoulder and right shoulder

        # no extra identifiers detected, reusing parameters... 
        motion_9.set_parameter(motion_8)
        # 'starting from the middle, not from the start' indicates insert_point to 0
        insert_point = 0
        motion_9.set_insert_point(0)
        
        # save motion
        db.save_motion(motion_9, "motion_9")
    '''
    )
    
    prompt_sequence.append("Let's start from scratch, with a new motion, motion_0. Ready?")
    prompt_sequence.append("Yes!")
    
    return prompt_sequence



if len(sys.argv) > 1 and sys.argv[1] == "chatbot":
    read_progprompt_0() # this reads 'FacialMotion' class
    read_progprompt("")
    get_incontext()
    while True:
        # import pdb;pdb.set_trace()
        user_input = input("You: ")
        
        print("Chatbot:", user_input)
        prompt = user_input

        prompt_sequence.append("# " + prompt + "\n")
        error_prompt_sequence = prompt_sequence

        c, r = query_model(prompt_sequence, error_prompt_sequence, 0)
        prompt_sequence.append(c)
        print(c)


