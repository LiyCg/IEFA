"""
Function's to get timing to insert keyframes
    when_bshp: 
"""
# from timing import when_bshp, at_global_moment, at_frame
# from speed import change_speed
from motion_io import load_motion, save_motion
from motion import FacialMotion

# relative_moments = ["min_range", "mid_range", "max_range"] # "0.0, 0.5, 1.0" of bshp parameter values

# global_moments = ["start_of_motion", "end_of_motion", "middle_of_motion"]

# intensities = ["low", "mid", "high"] # 0.25, 0.5, 0.75 -> how far from neutral

# # speeds = ["short", "medium", "long", "entire_motion"] # '1, 10, 30' frames

# speed_locations = ["front", "keyframes", "back"] # before keyframes / during keyframes / after keyframes


##########
## TODO ## 
## - (WIP) redo all below, regarding the creation of 'FacialMotion' class
## - redo in-context scenario
# def do_activate_emotion(motion):
#     assert(motion.cur_activated in full_blendshape_targets)
#     FACS_list = []
    
#     for e in emotion:
#         for FACS_unit in emotions[e]:    
#             FACS_list.append(FACS_unit)
#     do_activate_FACS(FACS_list, intensity, speed_location, time) 


# def do_activate_FACS(FACS_list, intensity, speed_location, time):
#     assert(FACS_list in FACS_units)
#     blendshape_target_list = []
#     for FACS_unit in FACS_list:
#         for bt in FACS_units[FACS_unit]:
#             blendshape_target_list.append(bt)
#     if len(time) == 2: # time should be an array with two elements
#         do_relative_activate_blendshape(blendshape_target_list, intensity, speed_location, time[0], time[1])
#     else:
#         do_activate_blendshape(blendshape_target_list,intensity, speed_location, time)


# """
# At specified time, insert interpt the keyframe animation 
# """
# def do_activate_blendshape(blendshape_target_list, intensity, speed_location,  time):
#     assert(blendshape_target_list in full_blendshape_targets)
#     assert((time >= 0.0 and time <= 1.0) or (time in global_moments))
#     activate_blendshape(blendshape_target_list, intensity, speed_location,  time)


# """
# At relative time, for example, to deal with timing of 'when mouth is max opened', insert the keyframe animation 
# """
# def do_relative_activate_blendshape(blendshape_target_list, intensity, speed_location, speed,  start_time, end_time):
#     assert(blendshape_target_list in full_blendshape_targets)
#     assert(start_time in relative_moments or (start_time >= 0.0 and start_time <= 1.0))
#     assert(end_time in relative_moments or (end_time >= 0.0 and end_time <= 1.0))
#     relative_activate_blendshape(blendshape_target_list, intensity, speed_location, start_time, end_time)


# """
# Changing the speed of keyframe animation.
# Increase or decrease 'speed' number of frames at 'speed_location'
# """
# def do_change_speed(speed_location, speed):
#     assert(speed_location in speed_locations)

#     change_speed(speed_location, speed)
    


################################# example codes from pre-prompt #################################

########################
## starting scenarios ##
########################

audio_path = ""
# the person is talking. Make a face when someone won a lottery in the middle of the sequence. 
def face_win_lottery():
    # initialize motion(only once for saving "motion_1")
    motion_1 = FacialMotion(audio_path) # loads 'FacialMotion' class instance, the first one is original speech animation motion 

    # the original motion is that the persion talking. 
    # the desired edit is to ativate surprised face in the middle of the motion

    # the involved identifiers are 'surprised' and 'happy'
    # no specification on intensity, so default 
    identifiers = ["surprised", "happy"]
    motion_1.activate_hierarchical(identifiers=identifiers)
    # no specification on speed_location, so default (initial value of 'FacialMotion' class)
    # 'in the middle of the sequence' means ratio 0.5 of 'insert_point'
    insert_point = 0.5
    motion_1.set_insert_point(insert_point)
    
    motion_1.activate_blendshape() # retrieves all set motion and converts into expression code to finally deconde into facial animation 
    
    # save motion
    motion_1.save_motion("motion_1")


audio_path = ""
# The person is talking. Add a brief cheek puff at frame 35. 
def cheeck_puff():
    # initialize motion
    motion_1 = FacialMotion(audio_path)
    
    # the original motion is that the persion talking. 
    # the desired edit is to ativate cheek puff at frame 35
    
    # the involved identifiers are 'cheek_blow'
    # no specification on intensity, so default 
    identifiers = ["cheek_blow"]
    motion_1.activate_hierarchical(identifiers=identifiers)
    # 'brief' indicates short in keyframe speed, so just half the default(=15)
    speed_locations = ["keyframes"]
    speed = [0.5]
    # ratio is specified so 'replace=False' (if number of frames are specified, 'replace=True')
    motion_1.set_speed(speed_locations, speed, replace=False)
    # frame 35 literally means at frame "35"
    insert_point = 35
    motion_1.set_insert_point(insert_point)
    
    motion_1.activate_blendshape()
    
    # save motion
    save_motion("motion_1")



# The person is talking. Add a sharp brow furrow and narrow the eyes starting from almost the end of the motion (to show deep concentration or seriousness). 
audio_path = ""
def brow_furrow_eye_narrow():
    # initialize motion=
    motion_1 = FacialMotion(audio_path)
    
    # the original motion is that the persion talking. 
    # the desired edit is to ativate eye squint face at almost the end
    
    # the involved identifiers are 
    # brow_lowerer for brow furrow
    # lid_tightener for narrowed eyes
    # sharp usually means high in intensity, so x1.5 the current value
    intensities = [1.5, 1.5]
    identifiers = ['brow_lowerer','lid_tightener']
    motion_1.activate_hierarchical(identifiers, intensities)   
    # no specification on speed_location, so default
    # 'almost the end' indicates roughly 0.7 in 'insert_point'
    insert_point = 0.7
    motion_1.set_insert_point(insert_point)
    
    motion_1.activate_blendshape()
    
    # save motion
    save_motion("motion_1")
    

audio_path = ""
# The person is talking. The person begins to smile with only left corner of mouth. 
def smile_left():
    # initialize motion=
    motion_1 = FacialMotion(audio_path)
    
    # the original motion is that the persion talking. 
    # The desired edit is to ativate smile only with left corner of mouth starting from the motion
    
    # the involved identifiers are 
    # mouth_smile_l
    # no specification on intensity, so default
    identifiers = ['mouth_smile_l']
    motion_1.activate_hierarchical(identifiers=identifiers)
    # no specification on speed_location, so default
    # 'begins to' indicates 0  in 'insert_point'
    insert_point = 0
    motion_1.set_insert_point(insert_point)
    
    motion_1.activate_blendshape()
    
    # save motion
    save_motion("motion_1")



# the person is talking with happy face starting from in middle of the sequence. Make happy face faster.
def smile_faster():
    # load previous motion
    motion_1 = load_motion("motion_1") # changing speed of some expression should mean the expression does exist, so assume there's been already a first editing.
    motion_2 = FacialMotion(animation_seq=motion_1.output_animation_seq)
    
    # the loaded motion is that the person is talking with happy face starting from in middle of the sequence. 
    # the desired edit is to make the happy face faster
    
    # no extra identifiers detected, reusing parameters... 
    motion_2.set_parameter(motion_1)
    # 'faster' indicates shorter number of frames in 'front_speed' frames in 'speed', so half the default
    speed_locations = ["front_speed"]
    speed = [0.5]
    motion_2.set_speed(speed_locations, speed)
    
    # everything else stays the same
    motion_2.activate_blendshape()

    # save motion
    save_motion(motion_2, "motion_2")
    

# The person is talking with closed eyes from the middle of the sequence. When eyes are fully shut, start smirking only on the left side  
def smile_left():
    # load previous motion
    motion_1 = load_motion("motion_1")
    motion_2 = FacialMotion(animation_seq=motion_1.output_animation_seq)
    
    # the loaded motion is that the person is talking with closed eyes from the middle of the sequence. 
    # The desired edit is when eyes are fully shut, start smirking only on the left side

    # the involved identifiers are 
    # mouth_smile_l
    # no specification on intensity, so default
    identifiers = ['mouth_smile_l']
    motion_2.activate_hierarchical(identifiers=identifiers)
    # no specification on speed_location, so default
    # 'when eyes are fully shut' indicates 'max' value for 'eye_blink_r' or 'eye_blink_l' (no specific eye specified)
    point = "eye_blink_r" 
    motion_2.set_insert_point(motion_2.when_bshp("max", point)) 
    
    motion_2.activate_blendshape() 
    
    # save motion 
    save_motion("motion_1") 