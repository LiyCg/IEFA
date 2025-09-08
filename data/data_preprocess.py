import face_alignment # if doesn't work, --> downgrade to this pip install opencv-python==4.5.5.64

"""
data generation code for TeTEC
"""

from skimage import io
import numpy as np
import torch 
import os
import cv2
from PIL import Image
# import mediapy as mp
import moviepy.editor as mp
import glob


def save_frames(video_path, root_path, frame_step=1):
    video_name =  os.path.basename(video_path).split('.')[0]
    output_path = os.path.join(root_path, video_name) # only extract video file's base name
    os.makedirs(output_path, exist_ok=True)
    
    # video = mp.read_video(video_path) # mediapy version
    video = mp.VideoFileClip(video_path) # moviepy version
    fps = video.fps
    
    # for i, frame in enumerate(video): # mediapy version
    for i, frame in enumerate(video.iter_frames()): # mediapy version
        i_to_str = str(i // frame_step).zfill(5)
        img = Image.fromarray(frame)
        img.save(os.path.join(output_path, f'{i_to_str}.png'))
    
    return fps

def make_video(frames_path, video_name, fps):
    # import pdb;pdb.set_trace()
    frames_path_list = sorted(glob.glob(os.path.join(frames_path, "*.png")))
    frames_img_list = [cv2.cvtColor(cv2.imread(frame_path), cv2.COLOR_BGR2RGB) for frame_path in frames_path_list]
    # mp.write_video(os.path.join(frames_path, f'{video_name}.mp4'), np.stack(frames_img_list, axis=0)) # mediapy verion
    clip = mp.ImageSequenceClip(frames_img_list, fps=fps)
    clip.write_videofile(os.path.join(frames_path, f'{video_name}.mp4'), codec='libx264')
    
    return 

def resize_img(input_dir, output_dir, image_size=(256,256)):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    # define image extensions 
    
    for img in [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg','.png','.jpeg'))]:
        img_path = os.path.join(input_dir, img)
        image = cv2.imread(img_path)
        
        resized = cv2.resize(image, image_size)
        
        output_path = os.path.join(output_dir, img)
        
        cv2.imwrite(output_path, resized)
    return 

def get_ldm(input_dir, output_dir):
    # Initialize face_alignment
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)

    # Get landmarks
    preds = fa.get_landmarks_from_directory(input_dir)

    # Create a directory for the output if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save each set of predictions to a separate .npy file
    for filename, landmarks in preds.items():
        if landmarks is not None:
            # Construct the output filename based on the input filename
            output_filename = os.path.join(output_dir, os.path.basename(filename).split('.')[0] + '.npy')
            # import pdb; pdb.set_trace()
            # Save the landmarks
            np.save(output_filename, landmarks[0])  # assuming you want to save the first set of landmarks

    print("Landmarks saved in '{}' directory.".format(output_dir))
    
def vis_ldm(image_dir, landmarks_dir):
    # Directories
    output_dir = landmarks_dir + 'landmarks_visualized/'  # Directory to save images with landmarks

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over the images in the image directory
    for image_file in [f for f in os.listdir(image_dir) if f.lower().endswith(('.png','.jpeg','.jpg'))]:
        image_path = os.path.join(image_dir, image_file)
        landmarks_path = os.path.join(landmarks_dir, os.path.splitext(image_file)[0] + '.npy')
        # import pdb; pdb.set_trace()
        
        # Check if the corresponding landmarks file exists
        if os.path.exists(landmarks_path):
            # Load image and landmarks
            image = cv2.imread(image_path)
            landmarks = np.load(landmarks_path)

            # Draw each landmark as a red dot
            for landmark in landmarks:
                x, y = int(landmark[0]), int(landmark[1])
                cv2.circle(image, (x, y), 2, (0, 0, 255), -1)

            # Save the annotated image
            output_filename = os.path.join(output_dir, os.path.basename(image_file))
            cv2.imwrite(output_filename, image)

    print(f"Images with landmarks saved in '{output_dir}' directory.")

def get_masked_face(input_directory, output_directory):
    
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, face_detector='sfd', device='cuda' if torch.cuda.is_available() else 'cpu')
    # Ensure output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # List all image files in the input directory
    image_path_list = sorted([img for img in os.listdir(input_directory) if img.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for image_path in image_path_list:
        image_full_path = os.path.join(input_directory, image_path)
        image = cv2.imread(image_full_path)
        # Detect facial landmarks
        preds = fa.get_landmarks(image)

        # Initialize a binary mask
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if preds is not None:
            # import pdb;pdb.set_trace()
            
            # Use only the first face detected (adjust if multiple faces are needed)
            face_landmarks = preds[0]
            
            # Select relevant landmark points to cover the entire face
            face_indices = list(range(17)) + list(range(17,21)) + list(range(22,27)) # Example: jawline points

            # Create a convex hull from the selected points
            hull = cv2.convexHull(np.array([face_landmarks[i] for i in face_indices], dtype=np.int32))

            # Draw the convex hull on the mask
            cv2.fillConvexPoly(mask, hull, 255)

        ## Save the binary mask as an image
        mask = mask.reshape(*mask.shape,1)
        np_mask_stacked = np.concatenate((mask,mask), axis=-1)
        np_mask_stacked = np.concatenate((np_mask_stacked, mask), axis=-1)
        np_masked_image = cv2.bitwise_and(image, np_mask_stacked) # bitwise_and is the key or will yield wrong image even if you convert it into BGR order or RGB order
        masked_image_path = os.path.join(output_directory, image_path)
        cv2.imwrite(masked_image_path, np_masked_image)
    print(f"saved done at {output_directory}")
        

if __name__ == '__main__':
    
    from tqdm import tqdm
    # input_dir = './Multiface_256/'
    # output_dir = './Multiface_256/'
    
    #################
    ## Video to frame
    # video_root_path = "/source/inyup/TeTEC/data/MEAD/m03"
    # video_root_path = "/source/inyup/TeTEC/data/MEAD/ffhq_align/m03/m03-crop-v1"
    # video_root_path = "/source/inyup/TeTEC/data/MEAD/ffhq_align/m03/m03-crop-v1/m03-angry-level_1-001/masked"
    video_root_path = "/source/inyup/TeTEC/data/MEAD/ffhq_align/m03/m03-crop-v1/m03-fear-level_3-001/masked"
    
    total_list = sorted(glob.glob(video_root_path + '/*'))

    video_path_list = sorted(glob.glob(video_root_path + "/*.mp4")) # 총 667개 video(original) / 총 665 개 in 'm03-crop-v1' / 총 664 개 in 'm03-crop'
    # video_path_list = sorted(glob.glob(video_root_path + "/m03-fear-level_3-001.mp4"))

    # i = 0
    # for video_path in tqdm(video_path_list):
    #     # import pdb;pdb.set_trace()
    #     if i == 0:
    #         fps = save_frames(video_path, video_root_path) # 30.0 fps for MEAD
    #         i += 1
    #     else: 
    #         save_frames(video_path,video_root_path)
    
    ####################
    ## Recon from frames 
        ## if you want to iterate over all lists of frame images path  
    # frames_path_list = [f for f in total_list if f not in video_path_list] 
    # for frames_path in frames_path_list: 
    #     video_name = frames_path + '_recon' 
    #     import pdb;pdb.set_trace() 
    #     make_video(frames_path, video_name, 30) 
    
        ## if you directly indicate where the frame images path is 
    # video_name = video_root_path.split('/')[-2] + '_masked' 
    # make_video(video_root_path, video_name, 30)
    
    #########################################
    ## Resizing multiface dataset to FFHQ size 
    # resize_img(input_dir, './Multiface_256')
    
    ###################################
    ## Remove everyhing other than face
    # get_masked_face('/source/inyup/TeTEC/data/MEAD/ffhq_align/m03/m03-crop-v1/m03-angry-level_1-001', '/source/inyup/TeTEC/data/MEAD/ffhq_align/m03/m03-crop-v1/m03-angry-level_1-001/masked')
    # get_masked_face('/source/inyup/TeTEC/data/MEAD/ffhq_align/m03/m03-crop-v1/m03-fear-level_3-001', '/source/inyup/TeTEC/data/MEAD/ffhq_align/m03/m03-crop-v1/m03-fear-level_3-001/masked')
    
    ###################
    ## Load pickle file
    # import pickle
    # pickle_file_path = "/source/inyup/TeTEC/data/MEAD/ffhq_align/m03/dataset_m003_vtx_dtw.pickle"
    # with open(file=pickle_file_path, mode='rb') as pickle_file:
    #     pf = pickle.load(pickle_file)
    # import pdb;pdb.set_trace()
    # print(pf.keys())
    
    
    