"""
This is FacialMotion class 
"""
# from AnySOTA.SpeechAnimation import SpeechAnimation # I don't know wht network to use yet
import pickle
import numpy as np

## biggest edit unit
emotions = {
    "angry" : ['cheek_raiser', 'lip_corner_puller'],
    "contempt" : ['lip_corner_puller', 'dimpler'],
    # "disgusted" : ['nose_wrinkler', 'lip_corner_depressor', 'lower_lip_depressor'], 
    "disgusted" : ['nose_wrinkler', 'upper_lip_raiser', 'lip_corner_depressor'], # my interpretation
    "fear" : ['inner_brow_raiser', 'outer_brow_raiser', 'upper_lid_raiser', 'lid_tightener', 'mouth_stretch'],
    "happy" : ["lip_corner_puller','cheek_raiser','lip_stretcher"],
    "sad" : ['inner_brow_raiser', 'brow_lowerer', 'lip_corner_depressor'],
    "surprised" : ['inner_brow_raiser', 'outer_brow_raiser','upper_lid_raiser','mouth_stretch']
}

## intermediate edit unit
FACS_units = {
    'inner_brow_raiser': ['brow_inner_up_l', 'brow_inner_up_r'],            # AU01
    'outer_brow_raiser': ['brow_outer_up_l', 'brow_outer_up_r'],            # AU02
    'brow_lowerer': ['brow_down_l', 'brow_down_r'],                         # AU04
    'cheek_raiser': ['cheek_squint_l', 'cheek_squint_r'],                   # AU06
    'upper_lid_raiser': ['eye_wide_l', 'eye_wide_r'],                       # AU05
    'lid_tightener': ['eye_squint_l', 'eye_squint_r'],                      # AU07
    'nose_wrinkler': ['nose_sneer_l', 'nose_sneer_r'],                      # AU09
    'upper_lip_raiser': ['mouth_shrug_upper'],                              # AU10
    'nasolabial_deepener': ['mouth_upper_up_l', 'mouth_upper_up_r'],        # AU11
    'lip_corner_puller': ['mouth_smile_l', 'mouth_smile_r'],                # AU12
    'dimpler': ['mouth_dimple_l', 'mouth_dimple_r'],                        # AU14
    'lip_corner_depressor': ['mouth_frown_l', 'mouth_frown_r'],             # AU15
    'lower_lip_depressor': ['mouth_lower_down_l', 'mouth_lower_down_r'],    # AU16
    'chin_raiser': ['mouth_shrug_lower'],                                   # AU17
    'lip_pucker': ['mouth_pucker'],                                         # AU18
    'lip_stretcher': ['mouth_stretch_l', 'mouth_stretch_r'],                # AU20
    'lip_funneler': ['mouth_funnel'],                                       # AU22
    'lip_presser': ['mouth_press_l', 'mouth_press_r'],                      # AU24
    'mouth_stretch': ['jaw_open'],                                          # AU27
    'lip_suck': ['mouth_roll_lower', 'mouth_roll_upper'],                   # AU28
    'jaw_thrust': ['jaw_forward'],                                          # AU29
    'jaw_sideways_left': ['jaw_left'],                                      # AU30 (left)
    'jaw_sideways_right': ['jaw_right'],                                    # AU30 (right)
    'cheek_blow': ['cheek_puff_l', 'cheek_puff_r'],                         # AU33
    'blink': ['eye_blink_l', 'eye_blink_r'],                                # AU45
    'eyes_turn_left': ['eye_look_out_l', 'eye_look_in_r'],                  # AU61
    'eyes_turn_right': ['eye_look_out_r', 'eye_look_in_l'],                 # AU62
    'eyes_up': ['eye_look_up_l', 'eye_look_up_r'],                          # AU63
    'eyes_down': ['eye_look_down_l', 'eye_look_down_r']                     # AU64
}

## smallest edit unit
full_blendshape_targets = [
    'brow_down_l',         # browDown_L
    'brow_down_r',         # browDown_R
    'brow_inner_up_l',     # browInnerUp_L
    'brow_inner_up_r',     # browInnerUp_R
    'brow_outer_up_l',     # browOuterUp_L
    'brow_outer_up_r',     # browOuterUp_R
    'cheek_puff_l',        # cheekPuff_L
    'cheek_puff_r',        # cheekPuff_R
    'cheek_squint_l',      # cheekSquint_L
    'cheek_squint_r',      # cheekSquint_R
    'eye_blink_l',         # eyeBlink_L
    'eye_blink_r',         # eyeBlink_R
    'eye_look_down_l',     # eyeLookDown_L
    'eye_look_down_r',     # eyeLookDown_R
    'eye_look_in_l',       # eyeLookIn_L
    'eye_look_in_r',       # eyeLookIn_R
    'eye_look_out_l',      # eyeLookOut_L
    'eye_look_out_r',      # eyeLookOut_R
    'eye_look_up_l',       # eyeLookUp_L
    'eye_look_up_r',       # eyeLookUp_R
    'eye_squint_l',        # eyeSquint_L
    'eye_squint_r',        # eyeSquint_R
    'eye_wide_l',          # eyeWide_L -> for eyes at max 
    'eye_wide_r',          # eyeWide_R -> for eyes at max
    'jaw_forward',         # jawForward
    'jaw_left',            # jawLeft
    'jaw_open',            # jawOpen -> for mouth at max
    'jaw_right',           # jawRight
    'mouth_close',         # mouthClose
    'mouth_dimple_l',      # mouthDimple_L
    'mouth_dimple_r',      # mouthDimple_R
    'mouth_frown_l',       # mouthFrown_L -> 입꼬리 at min
    'mouth_frown_r',       # mouthFrown_R -> 입꼬리 at min
    'mouth_funnel',        # mouthFunnel
    'mouth_left',          # mouthLeft
    'mouth_lower_down_l',  # mouthLowerDown_L
    'mouth_lower_down_r',  # mouthLowerDown_R
    'mouth_press_l',       # mouthPress_L
    'mouth_press_r',       # mouthPress_R
    'mouth_pucker',        # mouthPucker
    'mouth_right',         # mouthRight
    'mouth_roll_lower',    # mouthRollLower
    'mouth_roll_upper',    # mouthRollUpper
    'mouth_shrug_lower',   # mouthShrugLower
    'mouth_shrug_upper',   # mouthShrugUpper
    'mouth_smile_l',       # mouthSmile_L -> 입꼬리 at max
    'mouth_smile_r',       # mouthSmile_R -> 입꼬리 at max
    'mouth_stretch_l',     # mouthStretch_L 
    'mouth_stretch_r',     # mouthStretch_R
    'mouth_upper_up_l',    # mouthUpperUp_L
    'mouth_upper_up_r',    # mouthUpperUp_R
    'nose_sneer_l',        # noseSneer_L
    'nose_sneer_r'         # noseSneer_R
]

class FacialMotion():
    # def __init__(self, audio_path : str = None, animation_seq : np.array = None):
    def __init__(self, animation_seq : np.array = None):
        
        # self.audio_path = audio_path
        self.cur_activated = []
        # self.source_animation_seq = None
        if animation_seq:
            self.source_animation_seq = animation_seq
        else:
            self.source_animation_seq = np.zeros((110,53), dtype=np.float16)
        self.output_animation_seq = None
            
        ## get audio file (필요없을 수도)
        # if audio_path:
            # with open(self.audio_path, "rb") as af:
            #     self.audio = pickle.load(af)
            # self.source_animation_seq = SpeechAnimation(self.audio) # outputs Nx53 bshp sequence (not yet set)
        # else:
        #     if animation_seq:
        #         self.source_animation_seq = animation_seq
        #     else:
        #         raise ValueError("there's not animation_seq provided to initialize")
        
        ## get seq length
        self.seq_len = self.source_animation_seq.shape[0]
        
        ## set default parameters 
        self.key_idx = 0 
        self.intensity_vector = np.ones(53) * 0.5 # (53,) vector set to 0.5 as default
        self.key_exp_parameter = np.zeros(53) # (53,) vector set to 0.0 as default
        self.speed = {
            "front" : 7, 
            "keyframes" : 15,
            "back" : 7
        }

    def when_bshp(self, at : str = "max", point : str = None):
        
        assert at in ["max", "min"]
        
        idx = full_blendshape_targets.index(point)
        
        if at == "max":    
            max_val_frame_idx = 0
            max_val = -1000
            for i in range(self.seq_len): 
                frame_val = self.vertex_ani_seq[i][idx] # find corresponding bhsp targets' value
                if frame_val >= max_val:
                    max_val = frame_val 
                    max_val_frame_idx = i
            return_idx = max_val_frame_idx 
                
        elif at == "min":
            min_val_frame_idx = self.seq_len
            min_val = 1000
            for i in range(self.seq_len): 
                frame_val = self.vertex_ani_seq[i][idx]
                if frame_val <= min_val:
                    min_val = frame_val 
                    min_val_frame_idx = i    
            return_idx = min_val_frame_idx
            
        return return_idx 
        
    def set_activated(self, activated_list):
        """
        puts into cur_activated list(only blendshape targets not emotions and FACS)
        """
        for activated in activated_list:
            # checks if already activated 
            if activated in self.cur_activated:
                continue    
            else:
                self.cur_activated.append(activated)
    
    def set_insert_point(self, key_insert_point):
        if isinstance(key_insert_point, int):
            assert key_insert_point <= self.seq_len and key_insert_point >= 0, "key_insert_point frame is not within range"
            self.key_idx = key_insert_point
        else:
            assert key_insert_point <= 1.0 and key_insert_point >= 0.0, "key_insert_point ratio is not within range"
            self.key_idx = int(key_insert_point * self.seq_len)
    
    def set_speed(self, speed_locations : list, speed_frames : list, replace=False): 
        
        assert speed_locations in ["front","keyframes","back"], "unknown location to input speed_frames"
        for i, speed_location in enumerate(speed_locations):
            assert speed_frames[i] <= self.seq_len and speed_frames[i] >= 0, "speed_frames exceeds the allowable length"
            if replace:
                self.speed[speed_location] = speed_frames[i]
            else:
                self.speed[speed_location] = int(self.speed[speed_location] * speed_frames[i])
        
    def set_key_exp(self, intensities : list = None): 
        """
        after putting into cur_activated,
        get keyframe expression with given intensities
        """
        for i, c_a in enumerate(self.cur_activated):
            id = full_blendshape_targets.index(c_a)
            self.key_exp_parameter[id] = 1 # activate 
            if intensities == None:
                continue # leave it as default
            else:
                self.intensity_vector[id] = intensities[i] # set intensity for specified
            
        self.key_exp_parameter = self.key_exp_parameter * self.intensity_vector
        assert np.all(self.key_exp_parameter >= 0.0 and self.key_exp_parameter <= 1.0), "key_exp_parameter should be in the range [0.0,1.0]" 

    def set_parameter(self, motion):
        self.cur_activated = motion.cur_activated
        self.key_exp_parameter = motion.key_exp_parameter
        self.intensity_vector = motion.intensity_vector
        self.speed = motion.speed
        self.key_idx = motion.key_idx
        self.source_animation_seq = motion.source_animation_seq
        self.output_animation_seq = motion.output_animation_seq
        
    def set_direct_parameter(self, motion_info : dict):
        self.cur_activated = motion_info["cur_activated"]
        self.key_exp_parameter = motion_info["key_exp_parameter"]
        self.intensity_vector = motion_info["intensity_vector"]
        self.speed = motion_info["speed"]
        self.key_idx = motion_info["key_idx"]
        self.source_animation_seq = motion_info["source_animation_seq"]
        self.output_animation_seq = motion_info["output_animation_seq"]
        
    def activate_hierarchical(self, identifiers : list, intensities: list = None):
        """
        unified function to handle activation based on emotion, FACS, or direct blendshape target.
        """
        
        for identifier in identifiers:
            expanded_intensities = []
            if identifier in emotions.keys():
                facs_units = emotions[identifier]
                for facs in facs_units:
                    blendshape_targets = FACS_units[facs]
                    self.set_activated(blendshape_targets)  
                    if intensities:
                        for _ in range (len(blendshape_targets)):
                            expanded_intensities.append(intensities[identifiers.index(identifier)])

            elif identifier in FACS_units.keys():
                blendshape_targets = FACS_units[identifier]
                self.set_activated(blendshape_targets)
                if intensities:
                    for _ in range(len(blendshape_targets)):
                        expanded_intensities.append(intensities[identifiers.index(identifier)]) 

            elif identifier in full_blendshape_targets:
                self.set_activated([identifier])  
                if intensities:
                    expanded_intensities.append(intensities[identifiers.index(identifier)])
            
            else:
                raise ValueError("There's not such identifier in the current pipeline")
            
            self.set_key_exp(expanded_intensities if intensities else None)

    def insert_keyframe(self):
        """
        source animation sequence의 특정 point에 keyframe이(blendshape 파라미터로 이뤄진 특정 표정)이 삽입될 것  
        keyframe의 point: keyframe이 어디에 삽입되는지
        keyframe의 duration: keyframe이 몇 프레임에 걸쳐 있는지
        keyframe의 front speed: keyframe의 앞 frame들과 몇 프레임에 걸쳐 interpolate되는지 
        keyframe의 rear speed: keyframe의 뒷 frame들과 몇 프레임에 걸쳐 interpolate되는지 
        """
        output_sequence = self.source_animation_seq.copy()
        
        ## getting keyframe sequence the same size with source sequence
        keyframe_sequence = np.zeros_like(output_sequence)
        duration = self.speed["keyframes"]
        for i in range(duration):
            index = self.key_idx - int(duration / 2) + i
            if index < 0 or index >= len(self.seq_len):
                continue
            keyframe_sequence[index] = self.key_exp_parameter
                
        if self.key_idx + duration >= self.seq_len: # trial2 : duration could be 0~15, so now considering the back part
            end_idx = self.seq_len - 1
        else: 
            end_idx = self.key_idx + duration
        
        if self.key_idx - duration < 0: # trial2 : duration could be 0~15, so now considering the front part
            start_idx = 0
        else:
            start_idx = self.key_idx - duration
            if start_idx == self.seq_len: # if stitch_point is 1.0 and duration is 0 start_idx become out of index
                start_idx = self.seq_len - 1
        
        if start_idx == end_idx: # if they are the same
            keyframes = keyframe_sequence[start_idx]
        else:  
            keyframes = keyframe_sequence[start_idx : end_idx + 1] 
        
        num_keyframes = end_idx - start_idx + 1

        for i in range(num_keyframes): # trial2
            if start_idx + i >= self.seq_len:
                break
            output_sequence[start_idx + i] = keyframes[i]
        
        if start_idx - front_speed <= 0: # check if enough front_speed frames are available, if not, decrease front_speed to allowalbe length
            front_speed = start_idx - 1
            front_speed_frame = output_sequence[0]
        else:
            front_speed_frame = output_sequence[start_idx - (front_speed + 1)]
        
        for i in range(1, front_speed + 1):
            if start_idx - i < 0:
                break
            t = i / (front_speed + 1)
            output_sequence[start_idx - i] = t * (front_speed_frame) + (1-t) * (output_sequence[start_idx])
            
        if end_idx + back_speed >= self.seq_len - 1: # check if enough back_speed frames are available, if not, decrease back_speed to allowalbe length
            back_speed = self.seq_len - (end_idx - 1) - 1
            back_speed_frame = output_sequence[-1]
        else:
            back_speed_frame = output_sequence[end_idx + (back_speed + 1)]
        
        for i in range(1, back_speed + 1):
            if end_idx + i >= self.seq_len:
                break  
            t = i / (back_speed + 1)
            output_sequence[end_idx + i] = t * (back_speed_frame) + (1-t) * (output_sequence[end_idx])
            
        return output_sequence

    def activate_blendshape(self):
        """
        generate final edited blendshape sequence
        """
        self.output_animation_seq = self.insert_keyframe()
        


if __name__ == "__main__":
    
    audio_path = "./"
    motion_0 = FacialMotion(audio_path) 
    # import pdb;pdb.set_trace()
    print(motion_0.intensity_vector)