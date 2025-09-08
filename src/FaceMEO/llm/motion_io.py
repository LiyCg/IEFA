import sys
if __name__ != "main":
    sys.path.append("/source/inyup/IEFA/src/FaceMEO/llm/")
from motion import FacialMotion


class Motion_DB:
    def __init__(self):
        self.motion_database = {}   # {motion_name: {keyframes, curve_type}}
        self.motion_history = []    # 저장된 motion_name 순서대로 저장 (Undo용)

    def save_motion(self, motion: FacialMotion, motion_name: str):
        """
        Save the current motion state.
        """
        self.motion_database[motion_name] = {
            "keyframes": {frame: value.copy() for frame, value in motion.keyframes.items()},
            "curve_type": motion.curve_type
        }
        self.motion_history.append(motion_name)
        print(f"[Motion_DB] Saved '{motion_name}' successfully.")


    def load_motion(self, motion_name: str, return_dict=True):
        """
        Load a motion by name.
        """
        motion_info = self.motion_database.get(motion_name)
        if motion_info is None:
            raise ValueError(f"[Motion_DB] Motion '{motion_name}' not found.")
        
        if return_dict:
            return motion_info
        else:
            return FacialMotion(motion_info)


    def undo_last_motion(self):
        """
        Undo to the previous saved motion.
        Returns the loaded FacialMotion object.
        """
        if len(self.motion_history) < 2:
            raise ValueError("[Motion_DB] Cannot undo: no previous motion to revert to.")

        # 현재 마지막 motion 삭제
        last_motion = self.motion_history.pop()

        # 직전 motion 로드
        prev_motion = self.motion_history[-1]
        print(f"[Motion_DB] Undo: reverting to '{prev_motion}'.")

        return self.load_motion(prev_motion)


    def revert_to_motion(self, motion_name: str):
        """
        Revert directly to a specific snapshot by name.
        Also trims history so that reverted motion becomes the latest.
        """
        if motion_name not in self.motion_database:
            raise ValueError(f"[Motion_DB] Motion '{motion_name}' does not exist for revert.")

        # revert할 motion_name이 history에 존재하면 그 이후 기록 다 삭제
        if motion_name in self.motion_history:
            idx = self.motion_history.index(motion_name)
            self.motion_history = self.motion_history[:idx+1]
        else:
            # 만약 history에 없으면 새로 추가
            self.motion_history.append(motion_name)

        print(f"[Motion_DB] Reverted directly to '{motion_name}'.")
        return self.load_motion(motion_name)
    
    
    def get_latest_motion_name(self):
        """
        Returns the latest saved motion name.
        """
        if not self.motion_history:
            return None
        return self.motion_history[-1]

