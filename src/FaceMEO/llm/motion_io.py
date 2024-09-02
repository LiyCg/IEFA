"""
Holds motion database and io of motion
: load_motion, save_motion
"""
import sys
sys.path.append("./MEO/llm")
from motion import FacialMotion

class Motion_DB():
    
    def __init__(self):
    
        self.motion_database = {}

    def save_motion(self, motion : FacialMotion = None, motion_name = ""):
        """
        save the current state of the motion to the databse
        """
        self.motion_database[motion_name] = {
            "cur_activated" : motion.cur_activated.copy(),
            "key_exp_parameter" : motion.key_exp_parameter.copy(),
            "intensity_vector" : motion.intesity_vector.copy(),
            "speed" : motion.speed.copy(),
            "key_idx" : motion.key_idx.copy(),
            "source_animation_seq" : motion.source_animation_seq.copy(),
            "output_animation_seq" : motion.output_animation_seq.copy()
        }
        
        print(f"Saved motion {motion_name}") 



    def load_motion(self, motion_name, return_dict=False):
        """
        load the 'motion_name' specified from the database
        """
        motion_info = self.motion_database.get(motion_name, None)
        
        if motion_info:
            if return_dict: # just outputs dictionary
                return motion_info
            
            else: # just outputs FacialMotion class
                loaded_motion = FacialMotion() 
                loaded_motion.set_direct_parameter(motion_info)
                print(f"loaded motion {motion_name}\n")
                return loaded_motion 
        else:
            raise ValueError(f"{motion_name} doesn't exist in the database\n")
        
        