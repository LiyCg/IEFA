
## LOGS for 250119


## s3
"""
You: The person is talking. Add a slight surprised expression in the first quarter of the sequence at frame 25, with wide eyes (eye wide) and slightly raised inner eyebrows.
querying model
keyframe : [0.  0.  0.5 0.5 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.5 0.5 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]

Saved motion motion_1
Code successfully executed without error!
took 5.950298 seconds for current edit...
Chatbot: ```python
# initialize motion DB
db = Motion_DB()

def slight_surprise_first_quarter():
    # initialize motion
    motion_1 = FacialMotion()
    
    # the original motion is that the person is talking. 
    # the desired edit is to add a slight surprised expression in the first quarter of the sequence at frame 25, with wide eyes and slightly raised inner eyebrows.
    
    # adding new keyframe..
    # new identifiers detected! currently activated list should be replaced!
    motion_1.replace_activated = True
    # 'at frame 25' indicates insert_point to 25
    insert_point = 25
    motion_1.set_insert_point(insert_point)
    # involved identifiers are 'upper_lid_raiser' for wide eyes and 'inner_brow_raiser' for slightly raised inner eyebrows
    identifiers = ["upper_lid_raiser", "inner_brow_raiser"]
    # 'slight' indicates lower intensity, so 0.5x the default
    intensities = [0.5, 0.5]
    motion_1.activate_hierarchical(identifiers=identifiers, intensities=intensities)
    # no specification on speed_location, so default
    
    motion_1.activate_blendshape()
    
    # save motion
    db.save_motion(motion_1, "motion_1")
``` 

You: Make the surprised expression last until halfway through the sequence, then transition into a neutral expression at the end. Meaning you should make the keyframe duration till half the sequence. So half is 55 and the expression is at 25 so the length should be 30. Now go. 
Saved motion motion_2
Code successfully executed without error!
took 5.112371 seconds for current edit...
Chatbot: ```python
def surprise_to_neutral():
    # load motion
    motion_1 = db.load_motion("motion_1")
    motion_2 = FacialMotion(motion_1)

    # the original motion was that the person is talking and was edited to have a slight surprised expression in the first quarter of the sequence at frame 25, with wide eyes and slightly raised inner eyebrows.
    # the desired edit is to make the surprised expression last until halfway through the sequence, then transition into a neutral expression at the end.
    
    # changing existing keyframe..
    # existing identifier used. Leave it as it is. 
    motion_2.replace_activated = False
    # no specification on insert_point, so same as previous
    motion_2.set_insert_point(motion_1.key_idx)
    # set 'speed' to previous first
    # 'last until halfway through the sequence' indicates 30 number of frames for 'keyframes', replacing the value so replace=True
    motion_2.set_speed(list(motion_1.speed.keys()), list(motion_1.speed.values()), replace=True) # replace, not multiplication
    speed_locations = ["keyframes"]
    speed = [30]
    motion_2.set_speed(speed_locations, speed, replace=True)
    
    motion_2.activate_blendshape()

    # save motion
    db.save_motion(motion_2, "motion_2")
``` 


You: Now I want to make this surprised face apper more quickly. You should use the duration, intensity, identifier information of the motion_2 but insert point should be at frame 5. Now go.
Saved motion motion_3
Code successfully executed without error!
took 4.493752 seconds for current edit...
Chatbot: ```python
def surprise_quickly():
    # load motion
    motion_2 = db.load_motion("motion_2")
    motion_3 = FacialMotion(motion_2)

    # the original motion was that the person is talking and was edited to have a slight surprised expression in the first quarter of the sequence at frame 25, with wide eyes and slightly raised inner eyebrows, lasting until halfway through the sequence.
    # the desired edit is to make the surprised face appear more quickly, starting at frame 5.
    
    # changing existing keyframe..
    # existing identifier used. Leave it as it is. 
    motion_3.replace_activated = False
    # 'at frame 5' indicates insert_point to 5
    insert_point = 5
    motion_3.set_insert_point(insert_point)
    # no specification on speed_location, so same as previous
    motion_3.set_speed(list(motion_2.speed.keys()), list(motion_2.speed.values()), replace=True) # replace, not multiplication
    
    motion_3.activate_blendshape()

    # save motion
    db.save_motion(motion_3, "motion_3")
``` 

You: At about frame 80 lower the outer eyebrows and make any mouth related expression to add a hint of determination. Meaning you should guess what would be the best mouth related identifiers to make that kind of expression. Now go. 
keyframe : [0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0.]

Saved motion motion_4
Code successfully executed without error!
took 3.801098 seconds for current edit...
Chatbot: ```python
def determination_expression():
    # load motion
    motion_3 = db.load_motion("motion_3")
    motion_4 = FacialMotion(motion_3)

    # the original motion was that the person is talking and was edited to have a slight surprised expression in the first quarter of the sequence at frame 5, with wide eyes and slightly raised inner eyebrows, lasting until halfway through the sequence.
    # the desired edit is to lower the outer eyebrows and make a mouth-related expression to add a hint of determination at about frame 80.
    
    # adding new keyframe..
    # new identifiers detected! currently activated list should be replaced!
    motion_4.replace_activated = True
    # 'at about frame 80' indicates insert_point to 80
    insert_point = 80
    motion_4.set_insert_point(insert_point)
    # involved identifiers are 'outer_brow_raiser' for lowering outer eyebrows and 'lip_presser' for a determined mouth expression
    identifiers = ["outer_brow_raiser", "lip_presser"]
    # no specification on intensity, so default
    motion_4.activate_hierarchical(identifiers=identifiers)
    # no specification on speed_location, so default
    
    motion_4.activate_blendshape()

    # save motion
    db.save_motion(motion_4, "motion_4")
``` 

You: Initialize current motion (motion 5 you know) with no previous motion(like you did at motion_1, meaning we are starting over now) and Start with a neutral expression and transition to a sad expression by the middle of the sequence, with a frown (mouth frown) and slightly lowered inner eyebrows. It means that you should add the expression at the middle and the front speed should have the half of the sequence. But make the length about 40, which s a little bit shrorter than half.
querying model
keyframe : [0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0.]

Saved motion motion_5
Code successfully executed without error!
took 4.516335 seconds for current edit...
Chatbot: ```python
# initialize motion DB
db = Motion_DB()

def neutral_to_sad():
    # initialize motion
    motion_5 = FacialMotion()
    
    # the original motion is that the person is talking. 
    # the desired edit is to start with a neutral expression and transition to a sad expression by the middle of the sequence, with a frown and slightly lowered inner eyebrows.
    
    # adding new keyframe..
    # new identifiers detected! currently activated list should be replaced!
    motion_5.replace_activated = True
    # 'by the middle of the sequence' indicates insert_point to 0.5
    insert_point = 0.5
    motion_5.set_insert_point(insert_point)
    # involved identifiers are 'lip_corner_depressor' for a frown and 'inner_brow_raiser' for slightly lowered inner eyebrows
    identifiers = ["lip_corner_depressor", "inner_brow_raiser"]
    # no specification on intensity, so default
    motion_5.activate_hierarchical(identifiers=identifiers)
    # 'make the length about 40' indicates 40 number of frames for 'front' in 'speed', replacing the value so replace=True
    speed_locations = ["front"]
    speed = [40]
    motion_5.set_speed(speed_locations, speed, replace=True)
    
    motion_5.activate_blendshape()
    
    # save motion
    db.save_motion(motion_5, "motion_5")
``` 

"""



## s4
"""
You: Initialize current motion (motion 5 you know) with no previous motion(like you did at motion_1, meaning we are starting over now) and Start with a neutral expression and transition to a sad expression by the middle of the sequence, with a frown (mouth frown) and slightly lowered inner eyebrows. It means that you should add the expression at the middle and the front speed should have the half of the sequence. But make the length about 40, which s a little bit shrorter than half.
Saved motion motion_5
Code successfully executed without error!
took 4.516335 seconds for current edit...
Chatbot: ```python
# initialize motion DB
db = Motion_DB()

def neutral_to_sad():
    # initialize motion
    motion_5 = FacialMotion()
    
    # the original motion is that the person is talking. 
    # the desired edit is to start with a neutral expression and transition to a sad expression by the middle of the sequence, with a frown and slightly lowered inner eyebrows.
    
    # adding new keyframe..
    # new identifiers detected! currently activated list should be replaced!
    motion_5.replace_activated = True
    # 'by the middle of the sequence' indicates insert_point to 0.5
    insert_point = 0.5
    motion_5.set_insert_point(insert_point)
    # involved identifiers are 'lip_corner_depressor' for a frown and 'inner_brow_raiser' for slightly lowered inner eyebrows
    identifiers = ["lip_corner_depressor", "inner_brow_raiser"]
    # no specification on intensity, so default
    motion_5.activate_hierarchical(identifiers=identifiers)
    # 'make the length about 40' indicates 40 number of frames for 'front' in 'speed', replacing the value so replace=True
    speed_locations = ["front"]
    speed = [40]
    motion_5.set_speed(speed_locations, speed, replace=True)
    
    motion_5.activate_blendshape()
    
    # save motion
    db.save_motion(motion_5, "motion_5")
``` 


You: At the peak of the sad expression(meaning the point at 0.5 again), lower the inner brows to add a touch of vulnerability. 
Saved motion motion_6
Code successfully executed without error!
took 5.266529 seconds for current edit...
Chatbot: ```python
def add_vulnerability():
    # load motion
    motion_5 = db.load_motion("motion_5")
    motion_6 = FacialMotion(motion_5)

    # the original motion was that the person is talking and was edited to start with a neutral expression and transition to a sad expression by the middle of the sequence, with a frown and slightly lowered inner eyebrows.
    # the desired edit is to lower the inner brows at the peak of the sad expression to add a touch of vulnerability.
    
    # changing existing keyframe..
    # existing identifier used. Leave it as it is. 
    motion_6.replace_activated = False
    # 'at the peak of the sad expression' indicates insert_point to 0.5
    insert_point = 0.5
    motion_6.set_insert_point(insert_point)
    # involved identifiers are 'brow_lowerer' for lowering inner brows
    identifiers = ["brow_lowerer"]
    # no specification on intensity, so default
    motion_6.activate_hierarchical(identifiers=identifiers)
    # no specification on speed_location, so same as previous
    motion_6.set_speed(list(motion_5.speed.keys()), list(motion_5.speed.values()), replace=True) # replace, not multiplication
    
    motion_6.activate_blendshape()

    # save motion
    db.save_motion(motion_6, "motion_6")
``` 



You: Lower it more!
keyframe : [1.5 1.5 1.  1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1.  1.  0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0. ]

Saved motion motion_7
Code successfully executed without error!
took 5.392709 seconds for current edit...
Chatbot: ```python
def lower_inner_brows_more():
    # load motion
    motion_6 = db.load_motion("motion_6")
    motion_7 = FacialMotion(motion_6)

    # the original motion was that the person is talking and was edited to start with a neutral expression and transition to a sad expression by the middle of the sequence, with a frown and slightly lowered inner eyebrows. Then, lower the inner brows at the peak of the sad expression to add a touch of vulnerability.
    # the desired edit is to lower the inner brows more.
    
    # changing existing keyframe..
    # existing identifier used. Leave it as it is. 
    motion_7.replace_activated = False
    # no specification on insert_point, so same as previous
    motion_7.set_insert_point(motion_6.key_idx)
    # involved identifiers are 'brow_lowerer'
    identifiers = ["brow_lowerer"]
    # 'lower it more' indicates higher intensity, so x1.5 the current value, multiplying so replace=False
    intensities = [1.5]
    motion_7.activate_hierarchical(identifiers=identifiers, intensities=intensities, replace=False)
    # no specification on speed_location, so same as previous
    motion_7.set_speed(list(motion_6.speed.keys()), list(motion_6.speed.values()), replace=True) # replace, not multiplication
    
    motion_7.activate_blendshape()

    # save motion
    db.save_motion(motion_7, "motion_7")
``` 



You: Toward the end (last quarter), make the happy face.
keyframe : [0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1. 0.
 0. 0. 0. 0. 0.]

Saved motion motion_8
Code successfully executed without error!
took 4.450497 seconds for current edit...
Chatbot: ```python
def happy_towards_end():
    # load motion
    motion_7 = db.load_motion("motion_7")
    motion_8 = FacialMotion(motion_7)

    # the original motion was that the person is talking and was edited to start with a neutral expression and transition to a sad expression by the middle of the sequence, with a frown and slightly lowered inner eyebrows. Then, lower the inner brows at the peak of the sad expression to add a touch of vulnerability, and lower it more.
    # the desired edit is to make a happy face towards the end (last quarter).
    
    # adding new keyframe..
    # new identifiers detected! currently activated list should be replaced!
    motion_8.replace_activated = True
    # 'toward the end (last quarter)' indicates insert_point to 0.75
    insert_point = 0.75
    motion_8.set_insert_point(insert_point)
    # involved identifiers are 'happy'
    identifiers = ["happy"]
    # no specification on intensity, so default
    motion_8.activate_hierarchical(identifiers=identifiers)
    # no specification on speed_location, so default
    
    motion_8.activate_blendshape()

    # save motion
    db.save_motion(motion_8, "motion_8")
``` 


You: Also add sad face at that specific frame at previous motion.                 
keyframe : [1. 1. 1. 1. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 1.
 0. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 1. 1. 0.
 0. 0. 0. 0. 0.]

Saved motion motion_9
Code successfully executed without error!
took 6.177174 seconds for current edit...
Chatbot: ```python
def add_sad_at_specific_frame():
    # load motion
    motion_7 = db.load_motion("motion_7")
    motion_9 = FacialMotion(motion_7)

    # the original motion was that the person is talking and was edited to start with a neutral expression and transition to a sad expression by the middle of the sequence, with a frown and slightly lowered inner eyebrows. Then, lower the inner brows at the peak of the sad expression to add a touch of vulnerability, and lower it more.
    # the desired edit is to add a sad face at the specific frame (0.75) in the previous motion.
    
    # adding new keyframe..
    # new identifiers detected! currently activated list should be replaced!
    motion_9.replace_activated = True
    # 'at that specific frame' indicates insert_point to 0.75
    insert_point = 0.75
    motion_9.set_insert_point(insert_point)
    # involved identifiers are 'sad'
    identifiers = ["sad"]
    # no specification on intensity, so default
    motion_9.activate_hierarchical(identifiers=identifiers)
    # no specification on speed_location, so default
    
    motion_9.activate_blendshape()

    # save motion
    db.save_motion(motion_9, "motion_9")
``` 



You: Make it more sad to intensity of 1.6 maybe. 
keyframe : [1.6 1.6 1.6 1.6 0.  0.  0.  0.  1.  1.  0.  0.  0.  0.  0.  0.  0.  0.
 0.  0.  0.  0.  1.  1.  0.  0.  0.  0.  0.  0.  0.  1.6 1.6 0.  0.  0.
 0.  0.  0.  0.  0.  0.  0.  1.6 0.  1.  1.  0.  0.  0.  0.  0.  0. ]

Saved motion motion_10
Code successfully executed without error!
took 5.783414 seconds for current edit...
Chatbot: ```python
def increase_sad_intensity():
    # load motion
    motion_9 = db.load_motion("motion_9")
    motion_10 = FacialMotion(motion_9)

    # the original motion was that the person is talking and was edited to start with a neutral expression and transition to a sad expression by the middle of the sequence, with a frown and slightly lowered inner eyebrows. Then, lower the inner brows at the peak of the sad expression to add a touch of vulnerability, and lower it more. Also, add a sad face at the specific frame in the previous motion.
    # the desired edit is to make the sad expression more intense, to an intensity of 1.6.
    
    # changing existing keyframe..
    # existing identifier used. Leave it as it is. 
    motion_10.replace_activated = False
    # no specification on insert_point, so same as previous
    motion_10.set_insert_point(motion_9.key_idx)
    # involved identifiers are 'sad'
    identifiers = ["sad"]
    # 'intensity of 1.6' indicates setting intensity to 1.6
    intensities = [1.6]
    motion_10.activate_hierarchical(identifiers=identifiers, intensities=intensities, replace=True)
    # no specification on speed_location, so same as previous
    motion_10.set_speed(list(motion_9.speed.keys()), list(motion_9.speed.values()), replace=True) # replace, not multiplication
    
    motion_10.activate_blendshape()

    # save motion
    db.save_motion(motion_10, "motion_10")

"""


## s5 