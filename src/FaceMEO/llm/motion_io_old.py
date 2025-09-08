"""
Holds motion database and io of motion
: load_motion, save_motion
"""
import sys
if __name__ != "main":
    sys.path.append("/input/inyup/IEFA/src/FaceMEO/llm/")
from motion import FacialMotion

class Motion_DB():
    
    def __init__(self):
        self.motion_database = {}
        self.motion_database["so_far_activated"] = {}
    
    # def add_so_far_activated(self, activated_list : list = None, motion_name : str = ""):
    #     import pdb;pdb.set_trace()
    #     """
    #     add the current state of the motion to "so_far_activated" list
    #     """
    #     motions_so_far = list(self.motion_database["so_far_activated"].keys())
    #     self.motion_database["so_far_activated"][motion_name] = []
    #     for a in activated_list: # brow, mouth 
    #         for i, m in enumerate(motions_so_far): # motion_1, motion_2
    #             if a in self.motion_database["so_far_activated"][m]: # if exist in the list
    #                 continue # pass to next motion
    #             else: # if doesn't exist 
    #                 if i == (len(motions_so_far) - 1): # and it's the last motion you checked
    #                     break
    #                 else: # if it's not last motion, continue checking
    #                     continue
    #         self.motion_database["so_far_activated"][motion_name].append(a)   
    
    # def referesh_so_far_activated(self, motion_name : str = "", reset=True):
    #     """
    #     refresh the "so_far_activated" list
    #     """
    #     if reset:
    #         self.motion_database["so_far_activated"] = {}
    #     else:
    #         if motion_name == "":
    #             raise ValueError("refresh target doesn't exist")
    #         self.motion_database["so_far_activated"][motion_name] = 

    def save_motion(self, motion : FacialMotion = None, motion_name = ""):
        """
        save the current state of the motion to the databse
        """
        # import pdb;pdb.set_trace()
        self.motion_database[motion_name] = {
            "cur_activated" : motion.cur_activated.copy(),
            "key_exp_parameter" : motion.key_exp_parameter.copy(),
            "intensity_vector" : motion.intensity_vector.copy(),
            "speed" : motion.speed.copy(),
            "key_idx" : motion.key_idx,
            "source_animation_seq" : motion.source_animation_seq.copy(),
            "output_animation_seq" : motion.output_animation_seq.copy()
        }
        self.add_so_far_activated(motion.cur_activated, motion_name)
        print(f"Saved motion {motion_name}") 
        
        return self.motion_database 


    def load_motion(self, motion_name : str = "", return_dict=False):
        """
        load the 'motion_name' specified from the database
        """
        motion_info = self.motion_database.get(motion_name, None)
        import pdb;pdb.set_trace()
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
        
        