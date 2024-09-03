from motion_io import Motion_DB
from motion import FacialMotion

# initialize motion DB
db = Motion_DB()
# the person is talking. Make a face when someone won a lottery in the middle of the sequence. 
def face_win_lottery():
    # initialize motion(only once for saving "motion_1")
    motion_1 = FacialMotion() # loads 'FacialMotion' class instance, the first one is original speech animation motion 

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
    db.save_motion(motion_1, "motion_1")

# initialize motion DB
db = Motion_DB()
# The person is talking. Add a brief cheek puff at frame 35. 
def cheeck_puff():
    # initialize motion
    motion_1 = FacialMotion() 
    
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
    db.save_motion(motion_1, "motion_1")

# initialize motion DB
db = Motion_DB()
# The person is talking. Add a sharp brow furrow and narrow the eyes starting from almost the end of the motion (to show deep concentration or seriousness). 
def brow_furrow_eye_narrow():
    # initialize motion
    motion_1 = FacialMotion()
    
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
    db.save_motion(motion_1, "motion_1")

# initialize motion DB
db = Motion_DB()
# The person is talking. The person begins to smile with only left corner of mouth. 
def smile_left():
    # initialize motion
    motion_1 = FacialMotion()
    
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
    db.save_motion(motion_1, "motion_1")

# assume there's already initialized motion db
# the person is talking with happy face starting from in middle of the sequence. Make happy face faster.
def smile_faster():
    # load previous motion
    motion_1 = db.load_motion("motion_1") # changing speed of some expression should mean the expression does exist, so assume there's been already a first editing.
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
    db.save_motion(motion_2, "motion_2")

# initialize motion DB
db = Motion_DB()
# the person is talking with happy face starting from in middle of the sequence. Change back to neutral abrutly.
def back_to_neutral_faster():
    # load previous motion
    motion_1 = db.load_motion("motion_1") # changing speed of some expression should mean the expression does exist, so assume there's been already a first editing.
    motion_2 = FacialMotion(animation_seq=motion_1.output_animation_seq)
    
    # the loaded motion is that the person is talking with happy face starting from in middle of the sequence. 
    # the desired edit is to make the happy face faster
    
    # no extra identifiers detected, reusing parameters... 
    motion_2.set_parameter(motion_1)
    # 'faster' indicates shorter number of frames in 'front_speed' frames in 'speed', so half the default, multiplying so replace=False
    speed_locations = ["back"]
    speed = [0.5]
    motion_2.set_speed(speed_locations, speed, replace=False)
    
    # everything else stays the same
    motion_2.activate_blendshape()

    # save motion
    db.save_motion(motion_2, "motion_2")    

# initialize motion DB
db = Motion_DB()
# The person is talking with closed eyes from the middle of the sequence. When eyes are fully shut, start smirking only on the left side  
def smile_left():
    # load previous motion
    motion_1 = db.load_motion("motion_1")
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
    db.save_motion(motion_2, "motion_2") 

# initialize motion DB
db = Motion_DB()
# The person is talking. As you talk, widen your eyes more.    
def widen_eyes_more():
    # initialize motion
    motion_1 = FacialMotion()
    
    # the original motion is that the persion talking. 
    # The desired edit is as you talk, widen your eyes more.

    # the involved identifiers are 
    # upper_lid_raiser
    # 'widen more' usually mean higher on intensity, so 1.5x the current value
    identifiers = ['upper_lid_raiser']
    intensities = [1.5]
    motion_1.activate_hierarchical(identifiers, intensities)
    # no specification on speed_location, so default
    # 'As you talk' indicates sequence length for keyframe speed, so seq_len
        # 'As you talk' indicates 0 in 'insert_point', so default, value is ratio so replacing not multiplying, replace = True
    speed_locations = ["keyframes"]
    speed = [motion_1.seq_len]
    motion_1.set_speed(speed_locations, speed, replace=True)
    
    motion_1.activate_blendshape() 
    
    # save motion 
    db.save_motion(motion_1, "motion_1") 

# initialize motion DB
db = Motion_DB()
# The person is talking. Switch to a sad expression and pause it for a second in the middle of the sequence.
def sad_expression():
    # initialize motion
    motion_1 = FacialMotion()
    
    # the original motion is that the person talking. 
    # The desired edit is to activate a sad expression during a pause in the middle of the sequence.
    
    # the involved identifiers are 'sad'
    # no specification on intensity, so default
    identifiers = ['sad']
    motion_1.activate_hierarchical(identifiers=identifiers)
    # 'in the middle of the sequence' indicate the middle of the sequence, so set 'insert_point' to 0.5
    insert_point = 0.5
    motion_1.set_insert_point(insert_point)
    # 'pause for a second' indicates 30 number of frames in 'keyframes' in 'speed', value is number of frarmes so replacing, replace=True
    speed_locations = ["keyframes"]
    speed = [30]
    motion_1.set_speed(speed_locations, speed, replace=True)
    
    motion_1.activate_blendshape()
    
    # save motion
    db.save_motion(motion_1, "motion_1")

# initialize motion DB
db = Motion_DB()
# The person is talking. Add a surprised expression followed by relaxation towards the end of the sequence.
def surprise_then_relax():
    # initialize motion
    motion_1 = FacialMotion()
    
    # the original motion is that the person talking. 
    # The desired edit is to show a surprised expression followed by relaxation towards the end.
    
    # the involved identifiers are 'surprised'
    # no specification on intensity, so default
    identifiers = ['surprised']
    motion_1.activate_hierarchical(identifiers=identifiers)
    
    # 'towards the end' indicates roughly 0.75 in 'insert_point'
    insert_point = 0.75
    motion_1.set_insert_point(insert_point)
    
    motion_1.activate_blendshape()
    
    # save motion
    db.save_motion(motion_1, "motion_1")

# initialize motion DB
db = Motion_DB()
# The person is talking. Add a focused gaze by narrowing the eyes at frame 60.
def focused_gaze():
    # initialize motion
    motion_1 = FacialMotion()
    
    # the original motion is that the person talking. 
    # The desired edit is to narrow the eyes for a focused gaze at the start.
    
    # the involved identifiers are 'lid_tightener'
    # no specification on intensity, so default
    identifiers = ['lid_tightener']
    motion_1.activate_hierarchical(identifiers=identifiers)
    
    # 'at the start' indicates 0 in 'insert_point'
    insert_point = 60
    motion_1.set_insert_point(insert_point)
    
    motion_1.activate_blendshape()
    
    # save motion
    db.save_motion(motion_1, "motion_1")