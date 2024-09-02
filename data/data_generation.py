from IPython.display import display, Image, Audio

import cv2
import base64
import time
# from openai import OpenAI
import os
import pickle
from argparse import ArgumentParser
import glob
import random
import numpy as np
from numpy.linalg import norm
from dtw.dtw import dtw
from livelink_MEAD import face_model_io
from livelink_MEAD.util import extract_expression_parameter, _trim_ict_parameter, v_render_sequence_meshes, render_sequence_meshes, _render_sequence_meshes, mux_audio_video
# CLIENT = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as env var>"))

SEED = 0

emo_list = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
level_list = ['1', '2', '3']
actor = 'M003'
sentence_M003 = [
                ['001','001','028','001','001','001','001','001'], # Sentence A
                ['002','002','029','002','002','002','002','002'], # Sentence B
                ['021','020','018','021','021','031','021','021'], # ...
                ['022','021','019','022','022','032','022','022'],
                ['023','022','020','023','023','033','023','023'],
                ['024','023','021','024','024','034','024','024'],
                ['025','024','022','025','025','035','025','025'],
                ['026','025','023','026','026','036','026','026'],
                ['027','026','024','027','027','037','027','027'],
                ['028','027','025','028','028','038','028','028'],
                ['029','028','026','029','029','039','029','029'],
                ['030','029','027','030','030','040','030','030'],
                ['003','003','030','003','003','003','003','003']
            ]
num_sentence = len(sentence_M003) # 13

def synchronize_vectors(vectors_path1, vectors_path2, output_path, index_range = [24,50]):
    
    ## load 
    vectors1 = np.load(vectors_path1) # anchor
    vectors2 = np.load(vectors_path2) # target
    
    ## extract only lip and mouth related bshp parameters from ICT-FaceKit 
    ev1 = vectors1[:, index_range[0]:index_range[1]+1]
    ev2 = vectors2[:, index_range[0]:index_range[1]+1]
    
    ## dtw 
    _, _, _, path = dtw(ev2, ev1, dist=lambda x, y: norm(x - y, ord=1))    
    
    a = path[0]
    b = path[1]
    
    sync_vectors2 = vectors1.copy()

    ## synchronize the vectors based on the dtw paths
    for i in range (1, len(path[0])):
        sync_vectors2[b[i]] = vectors2[a[i]]
    
    ## save
    np.save(output_path, sync_vectors2)
    
    print(f"Sync done! shape of the first one(=anchor) : {vectors1.shape} / shape of the aligned(=target) : {sync_vectors2.shape}")
    
    return sync_vectors2


"""
Version that directly use np.array as input
"""
def _synchronize_vectors(vectors1, vectors2, output_path, index_range = [24,50]):
    
    ## extract only lip and mouth related bshp parameters from ICT-FaceKit 
    ev1 = vectors1[:, index_range[0]:index_range[1]+1]
    ev2 = vectors2[:, index_range[0]:index_range[1]+1]
    
    ## dtw 
    _, _, _, path = dtw(ev2, ev1, dist=lambda x, y: norm(x - y, ord=1))    
    
    a = path[0]
    b = path[1]
    
    sync_vectors2 = vectors1.copy()

    ## synchronize the vectors based on the dtw paths
    for i in range (1, len(path[0])):
        sync_vectors2[b[i]] = vectors2[a[i]]
    
    ## save
    np.save(output_path, sync_vectors2)
    
    print(f"Sync done! shape of the first one(=anchor) : {vectors1.shape} / shape of the aligned(=target) : {sync_vectors2.shape}")
    
    return sync_vectors2

def v_synchronize_vectors(vectors1, vectors2, output_path, index_list = []):
    
    ## extract only lip and mouth related vertices from ICT-FaceKit 
    ev1 = vectors1[:, index_list]
    ev2 = vectors2[:, index_list]
    
    ## dtw 
    _, _, _, path = dtw(ev2, ev1, dist=lambda x, y: norm(x - y, ord=1))    
    
    a = path[0]
    b = path[1]
    
    sync_vectors2 = vectors1.copy()

    ## synchronize the vectors based on the dtw paths
    for i in range (1, len(path[0])):
        sync_vectors2[b[i]] = vectors2[a[i]]
    
    ## save
    np.save(output_path, sync_vectors2)
    
    print(f"Sync done! shape of the first one(=anchor) : {vectors1.shape} / shape of the aligned(=target) : {sync_vectors2.shape}")
    
    return sync_vectors2

def generate_text(video_path, frame_step=1):
    
    ## read video
    # import pdb;pdb.set_trace()
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    frame_count = 0
    while video.isOpened():
        success, frame = video.read() # frame > single image
        if not success:
            break
        
        if frame_count % frame_step == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
            
        frame_count += 1
        
    video.release()
    print(len(base64Frames), "frames read.")
    
    ## sample frames from base64Frames
    """
    initializes a display output with a unique display_id. 
    The display function is used to create an empty display output.
    None is passed as the initial content, meaning the display starts with no content.
    display_id=True ensures that the display output has a unique identifier that allows it to be updated later
    """
    display_handle = display(None, display_id=True)
    for img in base64Frames:
        """
        img.encode("utf-8") converts the base64 string to a UTF-8 encoded byte string.
        base64.b64decode(...) decodes the base64 byte string back into binary image data.
        Image(data=...) creates an IPython.display.Image object from the binary image data.
        display_handle.update(...) updates the previously created display output (display_handle) with the new image. 
        This effectively replaces the content of the display with the new image, creating an animation effect as the loop progresses.
        """
        display_handle.update(Image(data=base64.b64decode(img.encode("utf-8"))))
        """
        This line pauses the loop for 0.025 seconds (25 milliseconds) before moving to the next iteration. 
        This delay controls the frame rate of the image updates, making the transition between images smoother and more viewable as an animation.
        """
        time.sleep(0.025)

    PROMPT_MESSAGES = [
        {
            "role" : "user",
            "content" : [
                "These are frames from a video that I want to upload. Generate a compelling description that I can upload along with the video.", # example
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::50]),
            ]
        }
    ]
    
    params = {
        "model" : "gpt-4o",
        "messages" : PROMPT_MESSAGES,
        "max_tokens" : 200
    }
    
    result = CLIENT.chat.completions.create(**params)
    print(result.choises[0].message.content) # sample 
    
    return result


## 'Naive stitching with no blending'
## creates 13 X 42 sequences (for default 'dataset_m003_vtx_dtw_nolevel.pickle' data)
def stitch_meshes(meshes_path, data_num=10, stitch_point=0.5):
    
    with open(meshes_path, 'rb') as vtx_file:
        vtx_data = pickle.load(vtx_file)
    
    stitched_vtx_data = {}
    stitched_meshes = []
    sentences = vtx_data.keys()
    # import pdb;pdb.set_trace()
    for sentence in sentences:
        # done = []
        emotions = vtx_data[sentence].keys()
        new_emotions = {}
        for _, emotion in enumerate(emotions):
             mesh1 = vtx_data[sentence][emotion]
            #  done.append(emotion)
            #  others = [key for key in vtx_data[sentence].keys() if key not in done]
             others = [key for key in vtx_data[sentence].keys() if key != emotion]
             for _, other in enumerate(others):            
                mesh2 = vtx_data[sentence][other]
                stitched_mesh = stitch_mesh(mesh1, mesh2, stitch_point=stitch_point)
                stitched_emotion = "{0}_{1}_M003_front_{2}_{3}_{4}".format(emotion.split("_")[0], emotion.split("_")[1], other.split('_')[0], other.split('_')[1], emotion.split('_')[-1])
                new_emotions[stitched_emotion] = stitched_mesh
        
        # import pdb;pdb.set_trace()    
        stitched_vtx_data[sentence] = new_emotions
        
        
    return stitched_vtx_data


## naive stitching
def stitch_mesh(mesh1, mesh2, stitch_point=0.5):
    
    n  = mesh1.shape[0]
    stitch_frame = int(n * stitch_point)
    stitched_mesh = np.concatenate((mesh1[:stitch_frame], mesh2[stitch_frame:]), axis=0)

    return stitched_mesh


## blendshape insertion
"""
source animation sequence의 특정 point에 keyframe이(blendshape 파라미터로 이뤄진 특정 표정)이 삽입될 것  
keyframe의 point: keyframe이 어디에 삽입되는지
keyframe의 duration: keyframe이 몇 프레임에 걸쳐 있는지
keyframe의 front speed: keyframe의 앞 frame들과 몇 프레임에 걸쳐 interpolate되는지 
keyframe의 rear speed: keyframe의 뒷 frame들과 몇 프레임에 걸쳐 interpolate되는지 
"""
def insert_keyframe(source_sequence_path, keyframe_sequence_path, neutral_sequence_path = "", intensity = 1.0, stitch_point=0.5, duration=1, front_speed=3, back_speed=3):
    # import pdb;pdb.set_trace()
    source_sequence = np.load(source_sequence_path) # should be source, (N,53)
    keyframe_sequence = np.load(keyframe_sequence_path) # should be a target but keyframe, (N,53)
    
    # if neutral_sequence_path or intensity != 1.0: # always does this
    neutral_sequence = np.load(neutral_sequence_path) if neutral_sequence_path else np.zeros_like(keyframe_sequence)
    keyframe_sequence = intensity * keyframe_sequence + (1 - intensity) * neutral_sequence
        
    ## sanity check
    # source_sequence = np.random.rand(100, 53)
    # keyframe = np.random.rand(1,53)
        ## custom expression 
    # keyframe = np.zeros((1,53))
    # keyframe[:, 45] = 1
    # keyframe[:, 46] = 1
    
    seq_len = source_sequence.shape[0]
    # half_duration = duration // 2 # 10 // 2 -> 5 
    
    # stitch_frame_idx = int(seq_len * stitch_point) - half_duration
    stitch_frame_idx = int(seq_len * stitch_point) # trial2 : it's a middle point in the duration window
    
    # if stitch_frame_idx < 0:
    #     stitch_frame_idx = 0
    
    # if stitch_frame_idx + duration > seq_len:
    #     end_idx = seq_len 
    # else:
    #     end_idx = stitch_frame_idx + duration
    
    if stitch_frame_idx + duration >= seq_len: # trial2 : duration could be 0~15, so now considering the back part
        end_idx = seq_len - 1
    else: 
        end_idx = stitch_frame_idx + duration
    
    if stitch_frame_idx - duration < 0: # trial2 : duration could be 0~15, so now considering the front part
        start_idx = 0
    else:
        start_idx = stitch_frame_idx - duration
        if start_idx == seq_len: # if stitch_point is 1.0 and duration is 0 start_idx become out of index
            start_idx = seq_len - 1
    
    if start_idx == end_idx: # if they are the same
        keyframes = keyframe_sequence[start_idx]
    else:  
        keyframes = keyframe_sequence[start_idx : end_idx + 1] 
    
    ## substituting source frame value to keyframe value
    # for i in range(duration):
    #     if stitch_frame_idx + i >= seq_len:
    #         break
    #     source_sequence[stitch_frame_idx + i] = keyframes[i]
    
    num_keyframes = end_idx - start_idx + 1
    # if (num_keyframes < duration * 2 + 1):
    #     import pdb;pdb.set_trace()
        
    for i in range(num_keyframes): # trial2
        if start_idx + i >= seq_len:
            break
        source_sequence[start_idx + i] = keyframes[i]
    
    ## interpolating starting from end of front_speed frame
    # for i in range(1, front_speed + 1):
    #     if stitch_frame_idx - i < 0:
    #         break
        
    #     if stitch_frame_idx - front_speed - 1 < 0:
    #         front_speed_frame = source_sequence[0]
    #     else:
    #         front_speed_frame = source_sequence[stitch_frame_idx - front_speed - 1]    
       
    #     t = i / (front_speed + 1)
    #     if stitch_frame_idx - i >= 0:
    #         source_sequence[stitch_frame_idx - i] = t * (front_speed_frame) + (1-t) * (keyframes[0])
    #     else:
    #         break
    
    if start_idx - front_speed <= 0: # trial2, check if enough front_speed frames are available, if not, decrease front_speed to allowalbe length
        front_speed = start_idx - 1
        front_speed_frame = source_sequence[0]
    else:
        front_speed_frame = source_sequence[start_idx - (front_speed + 1)]
    
    for i in range(1, front_speed + 1):
        if start_idx - i < 0:
            break
        t = i / (front_speed + 1)
        source_sequence[start_idx - i] = t * (front_speed_frame) + (1-t) * (source_sequence[start_idx])
        
    # for i in range(1, back_speed + 1):
    #     if stitch_frame_idx + duration > (seq_len - 1): # if duration is odd, stitch point is not exactly in the middle, leading to shorter back part of the duration
    #         break
        
    #     if stitch_frame_idx + duration + back_speed - 1 > (seq_len - 1): # if the final back_speed frame exceeds the seq_len, just make the last frame final back_speed frame
    #         back_speed_frame = source_sequence[-1]
    #     else:
    #         back_speed_frame = source_sequence[stitch_frame_idx + duration + back_speed - 1]
        
    #     t = i / (back_speed + 1)
    #     # import pdb;pdb.set_trace()
    #     # print(f"idx wrong: {stitch_frame_idx + duration + i - 1}")
    #     if stitch_frame_idx + duration + i - 1 < seq_len:
    #         source_sequence[stitch_frame_idx + duration + i - 1] = t * (back_speed_frame) + (1-t) * (keyframes[-1])
    #     else:
    #         break
    if end_idx + back_speed >= seq_len - 1: # trial2, check if enough back_speed frames are available, if not, decrease back_speed to allowalbe length
        back_speed = seq_len - (end_idx - 1) - 1
        back_speed_frame = source_sequence[-1]
    else:
        back_speed_frame = source_sequence[end_idx + (back_speed + 1)]
    
    for i in range(1, back_speed + 1):
        if end_idx + i >= seq_len:
            break  
        t = i / (back_speed + 1)
        source_sequence[end_idx + i] = t * (back_speed_frame) + (1-t) * (source_sequence[end_idx])
        
    print(f"generated inserted sequence shape of {source_sequence.shape}")   
    return source_sequence


def generate_dynamic_sequence(data_root_path = "./livelink_MEAD", output_root_path = "", output_path = "", num_samples = 1_0, render=False):
    
    ## get all the folders(not files) inside 'data_root_path_dir' using glob and sort (there will be N number folders and that corresponds to 'sentences')
    sentences_folders = sorted([d for d in glob.glob(os.path.join(data_root_path, "*")) if os.path.isdir(d)])

    if render:
        face_model = face_model_io.load_face_model('/source/inyup/TeTEC/ICT-FaceKit/FaceXModel')
    
    result_dic = {}
    random.seed(SEED)

    ## start for loop for the folders(sentences) above 
    for id, sentences_folder in enumerate(sentences_folders):
        ## insdie the loop, start for loop iterate over all (A, B) occurrences for M emotions (ex. if [a,b,c,d,e,f,g,h] -> (a,a), (a,b) ... (a,h), (b,a), (b,b)...(h,g),(h,h)), over the folders(emotions, M number of them) inside the folders(sentences)
            emotion_folders = sorted([d for d in glob.glob(os.path.join(sentences_folder, "*"))])
            sentence_result_dic = {}
            neutral_sequence_root_path = [d for d in glob.glob(os.path.join(sentences_folder, "*")) if d.split("/")[-1].split("_")[-3] == 'neutral'][0]
            neutral_sequence_path = os.path.join(neutral_sequence_root_path, "trim_extracted_ict_animation_dtw.npy")
            for i, emotion_folder_a in enumerate(emotion_folders):
                # emotion_a = os.path.basename(emotion_folder_a).split('_')[0] 
                # if id == 1 and emotion_a in ['angry','contempt']:
                #     print("proceeding...\n")
                #     continue
                # import pdb;pdb.set_trace()
                for j, emotion_folder_b in enumerate(emotion_folders):
                    
                    if emotion_folder_a == emotion_folder_b:
                        continue
                    emotion_pair_key = f"{os.path.basename(emotion_folder_a)}_{os.path.basename(emotion_folder_b)}"
                    
                    if render:
                        output_root_vid_path = os.path.join(output_root_path, os.path.basename(sentences_folder))
                        video_output_root_dir = os.path.join(output_root_vid_path, emotion_pair_key)
                        os.makedirs(video_output_root_dir, exist_ok=True)
                    parameters_list = []    
                    parameters_meta_list = []
                ## insdie the loop, start for loop for 'num_samples' number of times, 
                    for k in range(num_samples):
                        ## using SEED = 0, get random values for each parameters need for using 'insert_keyframe' function
                        source_sequence_path = os.path.join(emotion_folder_a, "trim_extracted_ict_animation_dtw.npy") # dtw의 anchor가 지금은 angry인데, angry는 '_dtw' 안붙어있어서 그냥 '_dtw' 붙여야한다
                        keyframe_sequence_path = os.path.join(emotion_folder_b, "trim_extracted_ict_animation_dtw.npy") 
                        seq_len = np.load(source_sequence_path).shape[0]
                        ## intensity : (0.0 ~ 1.0) / stitch_point : [0.0 ~ 1.0] / duration : [1 ~ seq_len] 
                        ## for intensity, you should sample from gaussian but mean is 1.0(should be rather most frquently sampled), but the distribution is very smoothened gaussian               
                        
                        # intensity = min(max(np.random.normal(loc=1.0, scale=0.15), 0.0), 1.0)
                        # intensity = random.uniform(0.0, 1.0) # if below 0.4, just similar to neutral
                        intensity = random.uniform(0.5, 1.0) 
                        ## for stitch_point, you should sample from gaussian but mean is 0.5(should be rather most frequent), but the distribution is very smoothened gaussian
                        # stitch_point = min(max(np.random.normal(loc=0.5, scale=0.1), 0.0), 1.0) # first trial
                        # stitch_point = random.uniform(0.0, 1.0)
                        stitch_point = random.randint(0,10) / 10
                        ## for duration, you should sample from gaussian but mean is 10(should be rather most frequent), but the distribution is very smoothened gaussian
                        # duration = max(int(np.random.normal(loc=10, scale=3)), 1)
                        # duration = min(duration, seq_len)
                        duration = random.randint(0,15)
                        # half_duration = duration // 2
                        
                        ## for front_speed : [5 to (((stitch_point * seq_len) - half_duration)], you should sample from gaussian but mean is 5(should be rather most frequent), but the distribution is very smoothened gaussian
                        # front_speed = max(int(np.random.normal(loc=5, scale=2)), 1)
                        front_speed = random.randint(1,10)
                        ## for back_speed : [5 to (seq_len - ((stitch_point * seq_len) + half_duration)], you should sample from gaussian but mean is 5(should be rather most frequent), but the distribution is very smoothened gaussian
                        # back_speed = max(int(np.random.normal(loc=5, scale=2)), 1)
                        back_speed = random.randint(1,10)
                        ## for front_speed/back_speed, 1/2 times for same value for both and 1/2 times for different values 
                        if random.choice([True, False]):
                            back_speed = front_speed
                        
                        ## use these sampled parameter to get inserted_keyframes using 'insert_keyframe' function
                        inserted_sequence = insert_keyframe(
                            source_sequence_path=source_sequence_path,
                            keyframe_sequence_path=keyframe_sequence_path,
                            neutral_sequence_path=neutral_sequence_path,
                            intensity=intensity,
                            stitch_point=stitch_point,
                            duration=duration,
                            front_speed=front_speed,
                            back_speed=back_speed
                        )
                        
                        ## save these parameter info(metadata) in a list 
                        parameters_meta = {
                            "intensity" : intensity,
                            "stich_point" : stitch_point, 
                            "duration" : duration, 
                            "front_speed" : front_speed, 
                            "back_speed" : back_speed, 
                        }     
                        parameters_meta_list.append(parameters_meta)
                        
                        if k % 4 == 0:
                        # # import pdb;pdb.set_trace()
                            print(f"[{k} / {num_samples}] saving parameters > intensity: {parameters_meta['intensity']} / stitch_point: {parameters_meta['stich_point']} / duration: {parameters_meta['duration']} / front_speed: {parameters_meta['front_speed']} / back_speed: {parameters_meta['back_speed']}")
                        
                        parameters_list.append(inserted_sequence) 
                        
                        ## if render == True
                        if render:
                            video_name = f"{k}_{intensity:.1f}_{stitch_point:.1f}_{duration}_{front_speed}_{back_speed}"
                            _render_sequence_meshes(
                                expression_parameters=inserted_sequence,
                                video_name=video_name,
                                show_angles=False,
                                mode="shade",
                                out_root_dir = video_output_root_dir,
                                face_model=face_model,
                                )    
                    ## save these list into a value and save as dictionary giving certain key with it
                    sentence_result_dic[emotion_pair_key] = {
                        "data" : parameters_list,
                        "metadata" : parameters_meta_list}
                
            ## save these dictionaries into values of another dictionary per sentences(keys)
            result_dic[str(id)] = sentence_result_dic 
    
    ## save this dictionary into a pickle file with given name 'output_path'
    if output_path:
        with open(output_path, "wb") as f:
            pickle.dump(result_dic, f)
                        
    ## return the dictionary
    return result_dic


def get_vtx_seq(face_model=None, source_vector=None, end_idx = 9409): # end_idx : full face area of ICT-FaceKit
    # assert(face_model != None and source_vector != None)
    vtx_seq = []
    for frame_idx, parameter in enumerate(source_vector):
        face_model.set_expression(parameter)
        face_model.deform_mesh()
        vs = face_model._deformed_vertices.copy() 
        vs = vs[0:end_idx,:].flatten()
        vtx_seq.append(vs)
    return np.array(vtx_seq)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--vtx_file", type=str, default='./feature/dataset_m003_vtx_dtw_nolevel.pickle', help="vtx file of the flame model (pickle)")
    parser.add_argument("--audio_file", type=str, default='./feature/dataset_m003_wav2vec.pickle', help="audio file of the flame model (pickle)")
    parser.add_argument("--output_vtx_path", type=str, default='./feature/stitched', help="output path for stitched mesh data")
    parser.add_argument("--stitched_video_path", type=str, default='/source/inyup/TeTEC/faceClip/data/mead/M003/video/front/fear/level/m03-neutral-level_1-010.mp4')
    
    args = parser.parse_args()    
    
    # generate_text(args.stitched_video_path)
    
    ######## MEAD : 감정 레벨당 30개 video 존재 <- 1 neutral & 3 angry/contempt/disgusted/fear/happy/sad/surprised
    #################
    ### vtx_data ####
    #################
    
    ### data check 
    # with open(args.vtx_file, 'rb') as vtx_file:
    #     vtx_data = pickle.load(vtx_file)        
    # print(vtx_data.keys())
    
        ## default data
    # print(vtx_data['M003_front_angry_1_005'].shape) # frame 다다름
        ## dataset_m003_vtx_dtw.pickle data 
    # import pdb;pdb.set_trace()
    # print(vtx_data.keys())
    # print(vtx_data['1'].keys())
    # print(len(vtx_data['1'].keys())) # 22 animation sequences <- 1 neutral & 3 angry/contempt/disgusted/fear/happy/sad/surprised
    # print(vtx_data['1']['angry_1_M003_front_angry_1_002'].shape) # (78, 15069) 
    # print(vtx_data['1']['angry_2_M003_front_angry_2_002'].shape) # (78, 15069) 
    # print(vtx_data['1']['angry_3_M003_front_angry_3_002'].shape) # (78, 15069) 
    
    ### stitch 3D animation data
    # stitched_vtx_data = stitch_meshes(args.vtx_file, 13)
    # file_name = args.vtx_file.split('/')[-1].split('.')[0]
    # output_file_name = file_name + '_stitched'
    # # import pdb;pdb.set_trace()
    # with open(args.vtx_file.replace(file_name, output_file_name), 'wb') as output_vtx_file:
    #     pickle.dump(stitched_vtx_data, output_vtx_file)
    
    #     ## sanity check!
    # print(stitched_vtx_data.keys())
    # keys_list = list(stitched_vtx_data.keys())
    # print(stitched_vtx_data[keys_list[0]].keys())
    
    ##################
    ### audio_data ###
    ##################
    
    ### data check 
    # with open(args.audio_file, 'rb') as audio_file:
    #     audio_data = pickle.load(audio_file)  
        
    """
    현재 second stage에서 mesh는 predicted된 past frame mesh sequences만 가지고 cross-attention 하기 때문에 audio는 dtw할 필요가 없었다 
    mesh-audio pair(동일 frame 수)일 필요가 없기때문이라는 말이다. 
    
    audio 학습시에 batch=1 이기때문에 shape이 동일하지 않아도 돌아간다. (n, 29) 니까 29만 맞으면 됨 
    이래서 학습 시간이 상대적으로 느린데, 이걸 batch 학습 가능하도록, audio file을 모두 같은 n으로 transform해보기 
    """
        ## default data
    # print(len(audio_data.keys())) # 667 audio sequences
    # print(audio_data['M003_front_angry_1_001'].shape) # (98, 29)
    # print(audio_data['M003_front_angry_1_002'].shape) # (78, 29)
    
        ## dataset_m003_wavform.pickle data 
    # print(audio_data['M003_front_angry_1_001'].shape) # (52320,)
    # print(audio_data['M003_front_angry_1_002'].shape) # (41600,)
    # print(audio_data['M003_front_angry_1_003'].shape) # (20800,)
    
    
    ####################
    ### insert bshps ###
    ####################
    # source_sequence_path = "/source/inyup/TeTEC/data/livelink_MEAD/20240730_MySlate_neutral_1_01/trim_extracted_ict_animation_dtw.npy"
    # keyframe_sequence_path = "/source/inyup/TeTEC/data/livelink_MEAD/20240730_MySlate_angry_3_01/trim_extracted_ict_animation.npy" 
    
    # inserted_sequence = insert_keyframe(source_sequence_path, keyframe_sequence_path, stitch_point=0.7, duration=5, front_speed=5, back_speed=5)
    # inserted_sequence = insert_keyframe(source_sequence_path, keyframe_sequence_path, stitch_point=0.7, duration=15, front_speed=5, back_speed=5)
    # inserted_sequence = insert_keyframe(source_sequence_path, keyframe_sequence_path, stitch_point=0.3, duration=15, front_speed=10, back_speed=10)
    # inserted_sequence = insert_keyframe(source_sequence_path, keyframe_sequence_path, stitch_point=0.4, duration=15, front_speed=7, back_speed=7)
    # inserted_sequence = insert_keyframe(source_sequence_path, keyframe_sequence_path, stitch_point=0.4, duration=30, front_speed=10, back_speed=10) # front/back 최소 10은되야 artifact처럼 안보이는듯
    # inserted_sequence = insert_keyframe(source_sequence_path, keyframe_sequence_path, stitch_point=0.4, duration=30, front_speed=3, back_speed=3) # front/back 최소 10은되야 artifact처럼 안보이는듯

    # output_root_path = "./generated_data"
    # os.makedirs(output_root_path, exist_ok=True)
    # video_name = 'inserted_neutral_1_01_angry_3_01_t1'
    # video_name = 'inserted_neutral_1_01_angry_3_01_t2'
    # video_name = 'inserted_neutral_1_01_angry_3_01_t3'
    # video_name = 'inserted_neutral_1_01_angry_3_01_t4'
    # video_name = 'inserted_neutral_1_01_angry_3_01_t5'
    # video_name = 'inserted_neutral_1_01_angry_3_01_t6'

    # output_path = f'./generated_data/{video_name}.npy'
    
    # np.save(output_path, inserted_sequence)
    
    
    
    ################################################
    ## preprocess all the captured data (trim + dtw)
    
    """
    livelink_capture_csv_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240731_MySlate_neutral_1_022/MySlate_11_iPhone_raw.csv'
    output_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240731_MySlate_neutral_1_022/extracted_ict_animation.npy'
    extract_expression_parameter(livelink_capture_csv_path, output_path)
    
    reference_video_path = r'/source/inyup/TeTEC/data/MEAD/ffhq_align/m03/m03-crop/m03-neutral-level_1-022.mp4' 
    trim_output_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240731_MySlate_neutral_1_022/trim_extracted_ict_animation.npy'
    trim_ict_parameter(reference_video_path, output_path, trim_output_path)
    
    trim_output_dtw_path = r'/source/inyup/TeTEC/data/generated_data/inserted_neutral_1_01_angry_3_01_t6.npy' # keyframe inserted test
    video_name = "shade_iserted_neutral_1_01_angry_3_01_t6"
    render_sequence_meshes(trim_output_dtw_path, video_name, mode="shade", show_angles=False) # if shaded render
    from IPython.display import Video
    Video(f"result/{video_name}.mp4")
    """
    from IPython.display import Video
    root_path = "/source/inyup/TeTEC/faceClip/data/livelink_MEAD"
    ref_root_path = "/source/inyup/TeTEC/faceClip/data/MEAD/m03/m03-crop-v1" # for trim
    sentence_path_list = [f for f in glob.glob(os.path.join(root_path, "*")) if os.path.isdir(f)] # sorts is not desirable
    face_model = face_model_io.load_face_model('/source/inyup/TeTEC/ICT-FaceKit/FaceXModel') ## uncomment only when rendering

    for sentence_path in sentence_path_list:
        emotion_path_list = [s for s in glob.glob(os.path.join(sentence_path, "*"))]
            ## neutral is the source of DTW
        # source_emotion_path = [e for e in emotion_path_list if os.path.basename(e).startswith("neutral")][0]
        # source_emotion_vector_path = os.path.join(source_emotion_path, "trim_extracted_ict_animation.npy")
            # shortest seq is the source of DTW
        # shortest = 10000
        # for e in emotion_path_list:
        #     tmp = np.load(os.path.join(e, "trim_extracted_ict_animation.npy"))
        #     if tmp.shape[0] < shortest:
        #         shortest = tmp.shape[0]
        #         source_emotion_path = e
        #         source_emotion_vector_path = os.path.join(e, "trim_extracted_ict_animation.npy")
            ## angry is the source of DTW
        source_emotion_path = [e for e in emotion_path_list if os.path.basename(e).startswith("angry")][0]
        source_emotion_vector_path = os.path.join(source_emotion_path, "trim_extracted_ict_animation.npy")
        
        for emotion_path in emotion_path_list:
            #######
            ## TRIM
            
            # if os.path.basename(emotion_path).startswith('contempt_3_022'):
            # if os.path.basename(emotion_path).startswith('disgusted_3_023'):
            # if os.path.basename(emotion_path).startswith('happy_3_028'):
            # if os.path.basename(emotion_path).startswith('neutral_1_038'):
            # if os.path.basename(emotion_path).startswith('happy_3_030'):
            # if os.path.basename(emotion_path).startswith('sad_3_030'):
            # if os.path.basename(emotion_path).startswith('surprised_3_003'):
            #     import pdb;pdb.set_trace()
            # else:
            #     continue
            
            # livelink_capture_csv_path = glob.glob(os.path.join(emotion_path, "MySlate_*_iPhone_raw.csv"))[0]
            # output_path = os.path.join(emotion_path, "extracted_ict_animation.npy")
            # expression_parameter = extract_expression_parameter(livelink_capture_csv_path, output_path)
            # emotion = os.path.basename(emotion_path).split('_')[0]
            # sentence_num = os.path.basename(emotion_path).split('_')[-1]
            # if emotion == 'neutral':
            #     reference_video_name = f"m03-{emotion}-level_1-{sentence_num}.mp4"
            # else:
            #     reference_video_name = f"m03-{emotion}-level_3-{sentence_num}.mp4"
            # reference_video_path = os.path.join(ref_root_path, reference_video_name)
            # trim_output_path = os.path.join(emotion_path, "trim_extracted_ict_animation.npy")
            # _trim_ict_parameter(reference_video_path, expression_parameter, trim_output_path)
            
            #######
            ## DTW
            
            # data sanity check
            # if os.path.basename(emotion_path).startswith('contempt_3_022'):
            # if os.path.basename(emotion_path).startswith('disgusted_3_023'):
            # if os.path.basename(emotion_path).startswith('happy_3_028'):
            # if os.path.basename(emotion_path).startswith('neutral_1_038'):
            # if os.path.basename(emotion_path).startswith('happy_3_030'):
            # if os.path.basename(emotion_path).startswith('sad_3_030'):
            # if os.path.basename(emotion_path).startswith('surprised_3_003'):
            #     import pdb;pdb.set_trace()
            # else:
            #     continue
            
            # if not os.path.basename(sentence_path).startswith('9'):
            #     continue
            # else:
            #     print("Going through sentence 9")
                
                ###################
                ## blendshape based 
            # if emotion_path == source_emotion_path:
            #     target_emotion_vector_path = os.path.join(source_emotion_path, "trim_extracted_ict_animation.npy")
            #     out_path = os.path.join(source_emotion_path, "trim_extracted_ict_animation_dtw.npy")
            #     np.save(out_path, np.load(target_emotion_vector_path))
            #     print(f"neutral(source) out_parameter shape : {out_parameter.shape}")
            #     continue
            # else:   
                
            #     target_emotion_vector_path = os.path.join(emotion_path, "trim_extracted_ict_animation.npy")
            #     out_path = os.path.join(emotion_path, "trim_extracted_ict_animation_dtw.npy")
            #     out_parameter = synchronize_vectors(source_emotion_vector_path, target_emotion_vector_path, out_path)
            # # import pdb;pdb.set_trace()
            # print(f"out_parameter shape : {out_parameter.shape}")
                
                ###############
                ## vertex based 
            # source_emotion_vector = np.load(source_emotion_vector_path)
            # target_emotion_vector_path = os.path.join(emotion_path, "trim_extracted_ict_animation.npy")
            # target_emotion_vector = np.load(target_emotion_vector_path)
            # source_vtx_seq = get_vtx_seq(face_model, source_emotion_vector) # get vertex seq for source
            # target_vtx_seq = get_vtx_seq(face_model, target_emotion_vector) # get vertex seq for target
            #     ## to get only lip related vertices
            # lip_vtx_indices = np.load('/source/inyup/TeTEC/faceClip/data/livelink_MEAD/ict_lip_vtx.npy')
            # flatten_lip_vtx_indices = []
            # for i in lip_vtx_indices:
            #     # import pdb;pdb.set_trace()
            #     if i*3 >= 28227:
            #         continue
            #     flatten_lip_vtx_indices.append(i*3)
            #     flatten_lip_vtx_indices.append(i*3+1)
            #     flatten_lip_vtx_indices.append(i*3+2)
            
            # if emotion_path == source_emotion_path:
            #     target_emotion_vector_path = os.path.join(source_emotion_path, "trim_extracted_ict_animation.npy")
            #     # out_path = os.path.join(source_emotion_path, "vtx_trim_extracted_ict_animation_dtw.npy")
            #     # out_path = os.path.join(source_emotion_path, "lipvtx_trim_extracted_ict_animation_dtw.npy")
            #     out_path = os.path.join(source_emotion_path, "a_lipvtx_trim_extracted_ict_animation_dtw.npy") # /w shortest seq as source

            #     np.save(out_path, source_vtx_seq)
            #     print(f"source_vtx_seq saved : {source_vtx_seq.shape}")
            # else:
            #     target_emotion_vector_path = os.path.join(emotion_path, "trim_extracted_ict_animation.npy")
            #     # out_path = os.path.join(emotion_path, "vtx_trim_extracted_ict_animation_dtw.npy")
            #     out_path = os.path.join(emotion_path, "a_lipvtx_trim_extracted_ict_animation_dtw.npy") # /w shortest seq as source
            #     # out_vtx_seq = _synchronize_vectors(source_vtx_seq, target_vtx_seq, out_path) # save inside the func
            #     out_vtx_seq = v_synchronize_vectors(source_vtx_seq, target_vtx_seq, out_path, flatten_lip_vtx_indices)
            #     print(f"target_vtx_seq saved : {out_vtx_seq.shape}")
            


            ################
            ## Render meshes
            
            # if os.path.basename(emotion_path).startswith('contempt_3_022'):
            # if os.path.basename(emotion_path).startswith('disgusted_3_023'):
            # if os.path.basename(emotion_path).startswith('happy_3_028'):
            # if os.path.basename(emotion_path).startswith('happy_3_030'):
            # if os.path.basename(emotion_path).startswith('sad_3_030'):
            # if os.path.basename(emotion_path).startswith('neutral_1_040'):
            # if os.path.basename(emotion_path).startswith('surprised_3_003'):
            #     import pdb;pdb.set_trace()
            # else:
            #     continue
    
            print(f"Rendering {os.path.basename(sentence_path)}/{os.path.basename(emotion_path)}\n")
            # if emotion_path != source_emotion_path: # if don't want to redner neutral exp, uncooment this and indent below
                
            # import pdb;pdb.set_trace()
                ## render with bhsp version
            # parameter_dtw_path = os.path.join(emotion_path, "trim_extracted_ict_animation_dtw.npy")
            # video_name = f"shade_trim_extracted_ict_animation_dtw"
            # render_sequence_meshes(parameter_dtw_path, video_name, mode="shade", show_angles=False, out_root_dir = emotion_path, face_model = face_model) # if shaded render
                ## render with vtx version
            # vtx_dtw_path = os.path.join(emotion_path, "vtx_trim_extracted_ict_animation_dtw.npy")
            # video_name = f"shade_vtx_trim_extracted_ict_animation_dtw"
            # vtx_dtw_path = os.path.join(emotion_path, "lipvtx_trim_extracted_ict_animation_dtw.npy")
            # video_name = f"shade_lipvtx_trim_extracted_ict_animation_dtw"
            # vtx_dtw_path = os.path.join(emotion_path, "s_lipvtx_trim_extracted_ict_animation_dtw.npy")
            # video_name = f"s_shade_lipvtx_trim_extracted_ict_animation_dtw"
            vtx_dtw_path = os.path.join(emotion_path, "a_lipvtx_trim_extracted_ict_animation_dtw.npy")
            video_name = f"a_shade_lipvtx_trim_extracted_ict_animation_dtw"
            v_render_sequence_meshes(vtx_dtw_path, video_name, mode="shade", show_angles=False, out_root_dir = emotion_path, face_model = face_model)
            Video(f"{emotion_path}/{video_name}.mp4")
                
    ###########################
    # mux audio and video
    # audio_path = "/source/inyup/TeTEC/data/MEAD/ffhq_align/m03/m03-audio/m03-angry-level_3-001.wav" 
    # audio_path = "/source/inyup/TeTEC/faceClip/data/test/audio/userstudy/m03-angry-level_3-003.wav"
    # video_path = "/source/inyup/TeTEC/data/livelink_MEAD/result/20240730_MySlate_angry_3_01_anchor.mp4" # anchor
    # video_path = "/source/inyup/TeTEC/faceClip/data/test/result/qualitative_render/IEFA/con_angry_exp_happy.mp4"
    # video_path = "/source/inyup/TeTEC/faceClip/data/test/result/qualitative_render/IEFA/con_angry_003_exp_happy_021.mp4"
    # result_path = "/source/inyup/TeTEC/faceClip/data/test/result/qualitative_render/IEFA/con_angry_exp_happy_mux.mp4"
    # result_path = "/source/inyup/TeTEC/faceClip/data/test/result/qualitative_render/IEFA/con_angry_003_exp_happy_021_mux.mp4"
    # mux_audio_video(audio_path, video_path, result_path)
    
    
    ###################################################
    ## random data generation for all the captured data 

    # import pickle
    # result_dic = {}
    # output_root_path = "./dynamic_livelink_MEAD"
    # os.makedirs(output_root_path, exist_ok=True)
    # data_pickle_file = "m003_dynamic_bshp_dtw.pickle"
    # output_path = os.path.join(output_root_path, data_pickle_file)
    # # result_dic = generate_dynamic_sequence(output_root_path = output_root_path, output_path = output_path, render=False)

    #############
    ## data check 
    # with open(output_path, 'rb') as f:
    #     dynamic_bshp_dtw_data = pickle.load(f)  
    # # import pdb;pdb.set_trace()
    # face_model = face_model_io.load_face_model('/source/inyup/TeTEC/ICT-FaceKit/FaceXModel')
    # ############################
    # ## data sanity check(render)
    # for _, (sentence_idx, full_data) in enumerate(dynamic_bshp_dtw_data.items()):
    #     # import pdb;pdb.set_trace()
    #     if sentence_idx in ['0','1','2','3','4','5','6','7','8','9']:
    #         print('passing..\n')
    #         continue
    #     for emotion_pair in full_data.keys():
    #         if sentence_idx == '10' and emotion_pair.split('_')[0] in ['angry','contempt','disgusted','fear', 'happy','neutral']:
    #             print('passing..\n')
    #             continue
    #         output_sentence_path = os.path.join(sentence_idx, emotion_pair)
    #         output_sentence_path = os.path.join(output_root_path, output_sentence_path)
    #         parameter = full_data[f'{emotion_pair}']['data'][0]
    #         parameter_meta = full_data[f'{emotion_pair}']['metadata'][0] # only get the first sample 
    #         video_name = f"{parameter_meta['intensity']:.1f}_{parameter_meta['stich_point']}_{parameter_meta['duration']}_{parameter_meta['front_speed']}_{parameter_meta['back_speed']}"
    #         _render_sequence_meshes(
    #             expression_parameters=parameter,
    #             video_name=video_name,
    #             show_angles=False,
    #             mode='shade',
    #             out_root_dir=output_sentence_path,
    #             face_model=face_model
    #         )