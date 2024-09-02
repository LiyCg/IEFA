import sys
import csv
import os
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation
from functools import partial

from glob import glob
from tqdm import tqdm

import openmesh as om 
sys.path.append('./livelink_MEAD/') # added to access face_model_io in data_generation.py file
import face_model_io

import pandas as pd
import numpy as np
import moviepy.editor as mp
from IPython.display import Video

sys.path.append('./../')
sys.path.append('/input/inyup/TeTEC/faceClip/data/')
from data_preprocess import save_frames


# Define the mapping between morph target indices and their corresponding names
morph_targets = [
    "browDown_L", "browDown_R", "browInnerUp_L", "browInnerUp_R", 
    "browOuterUp_L", "browOuterUp_R", "cheekPuff_L", "cheekPuff_R", 
    "cheekSquint_L", "cheekSquint_R", "eyeBlink_L", "eyeBlink_R", 
    "eyeLookDown_L", "eyeLookDown_R", "eyeLookIn_L", "eyeLookIn_R", 
    "eyeLookOut_L", "eyeLookOut_R", "eyeLookUp_L", "eyeLookUp_R", 
    "eyeSquint_L", "eyeSquint_R", "eyeWide_L", "eyeWide_R", 
    "jawForward", "jawLeft", "jawOpen", "jawRight", "mouthClose", 
    "mouthDimple_L", "mouthDimple_R", "mouthFrown_L", "mouthFrown_R", 
    "mouthFunnel", "mouthLeft", "mouthLowerDown_L", "mouthLowerDown_R", 
    "mouthPress_L", "mouthPress_R", "mouthPucker", "mouthRight", 
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", 
    "mouthSmile_L", "mouthSmile_R", "mouthStretch_L", "mouthStretch_R", 
    "mouthUpperUp_L", "mouthUpperUp_R", "noseSneer_L", "noseSneer_R"
]

# Define the mapping from CSV columns to morph target names
csv_to_morph = {
    "EyeBlinkLeft": "eyeBlink_L", "EyeBlinkRight": "eyeBlink_R",
    "EyeLookDownLeft": "eyeLookDown_L", "EyeLookDownRight": "eyeLookDown_R",
    "EyeLookInLeft": "eyeLookIn_L", "EyeLookInRight": "eyeLookIn_R",
    "EyeLookOutLeft": "eyeLookOut_L", "EyeLookOutRight": "eyeLookOut_R",
    "EyeLookUpLeft": "eyeLookUp_L", "EyeLookUpRight": "eyeLookUp_R",
    "EyeSquintLeft": "eyeSquint_L", "EyeSquintRight": "eyeSquint_R",
    "EyeWideLeft": "eyeWide_L", "EyeWideRight": "eyeWide_R",
    "JawForward": "jawForward", "JawRight": "jawRight", "JawLeft": "jawLeft",
    "JawOpen": "jawOpen", "MouthClose": "mouthClose", "MouthFunnel": "mouthFunnel",
    "MouthPucker": "mouthPucker", "MouthRight": "mouthRight", "MouthLeft": "mouthLeft",
    "MouthSmileLeft": "mouthSmile_L", "MouthSmileRight": "mouthSmile_R",
    "MouthFrownLeft": "mouthFrown_L", "MouthFrownRight": "mouthFrown_R",
    "MouthDimpleLeft": "mouthDimple_L", "MouthDimpleRight": "mouthDimple_R",
    "MouthStretchLeft": "mouthStretch_L", "MouthStretchRight": "mouthStretch_R",
    "MouthRollLower": "mouthRollLower", "MouthRollUpper": "mouthRollUpper",
    "MouthShrugLower": "mouthShrugLower", "MouthShrugUpper": "mouthShrugUpper",
    "MouthPressLeft": "mouthPress_L", "MouthPressRight": "mouthPress_R",
    "MouthLowerDownLeft": "mouthLowerDown_L", "MouthLowerDownRight": "mouthLowerDown_R",
    "MouthUpperUpLeft": "mouthUpperUp_L", "MouthUpperUpRight": "mouthUpperUp_R",
    "BrowDownLeft": "browDown_L", "BrowDownRight": "browDown_R",
    "BrowInnerUp": "browInnerUp_L",  # Assuming both brows move together for inner up
    "BrowOuterUpLeft": "browOuterUp_L", "BrowOuterUpRight": "browOuterUp_R",
    "CheekPuff": "cheekPuff_L", "CheekPuff": "cheekPuff_R",  # Assuming both cheeks puff together
    "CheekSquintLeft": "cheekSquint_L", "CheekSquintRight": "cheekSquint_R",
    "NoseSneerLeft": "noseSneer_L", "NoseSneerRight": "noseSneer_R"
}


def extract_expression_parameter(livelink_capture_csv_path = "", output_path = ""):

    # Read the CSV file
    df = pd.read_csv(livelink_capture_csv_path)

    # import pdb;pdb.set_trace()
    # Extract the timecode and blendshape columns (assuming timecode is the first column)
    timecode_column = df.columns[0]
    blendshape_columns = df.columns[2:]  # Adjust based on actual CSV structure, 61 number of columns

    # Initialize a list to store the expression parameter vectors
    expression_vectors = [] # animation sequences

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        expression_vector = np.zeros((len(morph_targets))) # 53 morph targets
        for csv_col, morph_name in csv_to_morph.items():
            if morph_name in morph_targets:
                target_index = morph_targets.index(morph_name)
                expression_vector[target_index] = row[csv_col]
        expression_vectors.append(expression_vector)

    expression_vectors = np.array(expression_vectors) # convert to np araay
    
    print(f"Extracted from {os.path.basename(livelink_capture_csv_path)} has shape of {expression_vectors.shape}")
    # print(expression_vectors)
    np.save(output_path, expression_vectors)
    
    return expression_vectors


"""
version just gives vertices
"""
def v_render_sequence_meshes(Vs_path : np.array = None, 
                           video_name="",
                           bg_black=False,
                           fps=30,
                           face_only=True,
                           remove_axis=True,
                           show_angles=False,
                           figsize=(6,6),
                           mode="shade",
                           light_dir=np.array([0,0,1]),
                           linewidth = 1,
                           out_root_dir = "./result",
                           face_model = None,
                           ):
    
    # expression_parameters = np.load(expression_parameters_path)
    if face_model == None:
        face_model = face_model_io.load_face_model('/source/inyup/TeTEC/ICT-FaceKit/FaceXModel')
    
    frame_vertices = [] # [N, 53]
    Vs = np.load(Vs_path)
    frame_len = Vs.shape[0]
    Vs = Vs.reshape(frame_len,-1,3)
    # import pdb;pdb.set_trace()
    for frame_idx in range(frame_len):
        # # blendshape 53
        parameter = np.zeros((53))
        face_model.set_expression(parameter)
        # # Deform the mesh
        face_model.deform_mesh()
        # import pdb;pdb.set_trace()
        
        deformed_vertices = face_model._deformed_vertices.copy()
        deformed_vertices[:9409,] = Vs[frame_idx][:,:]
        frame_vertices.append(deformed_vertices)

    if bg_black:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    if show_angles:
        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(131, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
        ax2 = fig.add_subplot(132, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
        ax3 = fig.add_subplot(133, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
        if remove_axis:
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
    else:
        fig = plt.figure(figsize=figsize)
        _r = figsize[0] / figsize[1]
        fig_xlim = [-_r,_r]
        fig_ylim = [-1,+1]
        single_ax = fig.add_axes([0,0,1,1], xlim=fig_xlim, ylim=fig_ylim, aspect=1, frameon=False)
        if remove_axis:
            single_ax.axis('off')

    if face_only:
        ## face only
        v_idx, f_idx = 9409, 9230
    else:
        ## face + head + neck
        v_idx, f_idx = 11248, 11144

    # This is quad mesh!
    quad_F  = face_model._generic_neutral_mesh.face_vertex_indices()[:f_idx] # [N,4]
    tri_faces_1 = quad_F[:, [0, 1, 2]] # [N,3]
    tri_faces_2 = quad_F[:, [0, 2, 3]] # [N,3]
    F = np.vstack([tri_faces_1, tri_faces_2]) # [2N,3]
    Vs = frame_vertices # [V,3]

    # Pre-calculated transformation matrices for three views
    if mode == "mesh":
        proj = perspective(25, 1, 1, 100) # for mesh
        model = translate(0, 0, -2.5) 
    else:
        proj = ortho(-12,12,-12,12,1,100) # for shade
        model = translate(0, 0, -2.5) 
        
    if show_angles: # if wanna render with different angles, this should be coupled with 'mesh' mode
        MVPs = [
            proj @ model,
            proj @ model @ yrotate(-30),
            proj @ model @ yrotate(-90)
        ]
    else: # if only front
        MVP = proj @ model,
        MVP = MVP[0] # IDK, but it is tupled
        # import pdb;pdb.set_trace() 
    
    def render_mesh(ax, V, MVP, F):
        # quad to triangle
        # import pdb;pdb.set_trace()
        if mode == "mesh":
            VF = transform_vertices(V[:v_idx] , MVP, F, norm=True)
        else:
            VF = transform_vertices(V[:v_idx] , MVP, F, norm=False)
        
        T = VF[:, :, :2]
        Z = -VF[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)
        
        if mode == "shade":
            C = calc_face_norm(V, F) @ model[:3,:3].T # [3,3] -----> contains the transformed face normals
            I = np.argsort(Z) # -----------------------------------> depth sorting 
            T, C = T[I, :], C[I, :] # ensures that triangles are rendered from back to front
            
            NI = np.argwhere(C[:, 2] > 0).squeeze() # -------------> culling w/ normal,  checks the z-component of the normals, if  positive, the face is facing towards the camera.
            T, C = T[NI, :], C[NI, :] # only extracts faces facing front 
            
            C = np.clip((C @ light_dir), 0, 1) # ------------------> cliping range 0 - 1, resulting in light intensity value
            C = C[:, np.newaxis].repeat(3, axis=-1) # making RGB channels
            C = np.clip(C, 0, 1) # intensity values remain within the range [0, 1]
            C = C*0.6+0.15 # reduces the intensity by 40%, ensuring brightest are not too bright / adds a small base value to the intensity, ensuring that  darkest have some brightness
            collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        else:
            C = plt.get_cmap("gray")(Z)
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]
            
            NI = np.argwhere(C[:,2] > 0).squeeze()
            T, C  = T[NI, :], C[NI, :]
            collection = PolyCollection(T, closed=False, linewidth=0.23, facecolor=C, edgecolor="black")
        
        ax.add_collection(collection)

    def update(V):
        # Cleanup previous collections
        if show_angles:
            for _ax in [ax1, ax2, ax3]:
                for coll in _ax.collections:
                    coll.remove()
            
            # Render meshes for all views
            for _ax, mvp in zip([ax1, ax2, ax3], MVPs):
                render_mesh(_ax, V, mvp, F)
                
            return ax1.collections + ax2.collections + ax3.collections
        
        else:
            for coll in single_ax.collections:
                coll.remove()
            # import pdb;pdb.set_trace()
            render_mesh(single_ax, V, MVP, F)
            
            return single_ax.collections

    ani = FuncAnimation(fig, update, frames=tqdm(Vs, desc="Rendering frames", ncols=100), blit=True) # Vs[56:202]

    ### can be saved in difference format
    os.makedirs(out_root_dir, exist_ok=True)
    ani.save(f'{out_root_dir}/{video_name}.mp4', writer='ffmpeg', fps=fps)
    plt.close()


"""
version just gives numpy array, can render images
"""
def _render_sequence_meshes(expression_parameters = None, 
                           video_name="",
                           bg_black=False,
                           fps=30,
                           face_only=True,
                           remove_axis=True,
                           show_angles=True,
                           figsize=(6,6),
                           mode="shade",
                           light_dir=np.array([0,0,1]),
                           linewidth = 1,
                           out_root_dir = "./result",
                           face_model = None,
                           render_images=False,
                           ):
    
    # expression_parameters = np.load(expression_parameters_path)
    if face_model == None:
        face_model = face_model_io.load_face_model('/input/inyup/TeTEC/ICT-FaceKit/FaceXModel')
    
    frame_vertices = [] # [N, 53]
    
    for frame_idx, parameter in enumerate(expression_parameters):
        # # blendshape 53
        face_model.set_expression(parameter)
        # # Deform the mesh
        face_model.deform_mesh()
        # # Write the deformed mesh
        # face_model_io.write_deformed_mesh('/data/sihun/arkit_CSH/sample_identity_arkit_frame_{:06d}.obj'.format(idx), face_model)
        # # om.write_mesh(file_path, face_model._deformed_mesh, halfedge_tex_coord = True)
        frame_vertices.append(face_model._deformed_vertices.copy())


    if bg_black:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    if show_angles:
        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(131, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
        ax2 = fig.add_subplot(132, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
        ax3 = fig.add_subplot(133, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
        if remove_axis:
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
    else:
        fig = plt.figure(figsize=figsize)
        _r = figsize[0] / figsize[1]
        fig_xlim = [-_r,_r]
        fig_ylim = [-1,+1]
        single_ax = fig.add_axes([0,0,1,1], xlim=fig_xlim, ylim=fig_ylim, aspect=1, frameon=False)
        if remove_axis:
            single_ax.axis('off')

    if face_only:
        ## face only
        v_idx, f_idx = 9409, 9230
    else:
        ## face + head + neck
        v_idx, f_idx = 11248, 11144

    # This is quad mesh!
    quad_F  = face_model._generic_neutral_mesh.face_vertex_indices()[:f_idx] # [N,4]
    tri_faces_1 = quad_F[:, [0, 1, 2]] # [N,3]
    tri_faces_2 = quad_F[:, [0, 2, 3]] # [N,3]
    F = np.vstack([tri_faces_1, tri_faces_2]) # [2N,3]
    Vs = frame_vertices # [V,3]

    # Pre-calculated transformation matrices for three views
    if mode == "mesh":
        proj = perspective(25, 1, 1, 100) # for mesh
        model = translate(0, 0, -2.5) 
    else:
        proj = ortho(-12,12,-12,12,1,100) # for shade
        model = translate(0, 0, -2.5) 
        
    if show_angles: # if wanna render with different angles, this should be coupled with 'mesh' mode
        MVPs = [
            proj @ model,
            proj @ model @ yrotate(-30),
            proj @ model @ yrotate(-90)
        ]
    else: # if only front
        MVP = proj @ model,
        MVP = MVP[0] # IDK, but it is tupled
        # import pdb;pdb.set_trace() 
    
    def render_mesh(ax, V, MVP, F):
        # quad to triangle
        # import pdb;pdb.set_trace()
        if mode == "mesh":
            VF = transform_vertices(V[:v_idx] , MVP, F, norm=True)
        else:
            VF = transform_vertices(V[:v_idx] , MVP, F, norm=False)
        
        T = VF[:, :, :2]
        Z = -VF[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)
        
        if mode == "shade":
            C = calc_face_norm(V, F) @ model[:3,:3].T # [3,3] -----> contains the transformed face normals
            I = np.argsort(Z) # -----------------------------------> depth sorting 
            T, C = T[I, :], C[I, :] # ensures that triangles are rendered from back to front
            
            NI = np.argwhere(C[:, 2] > 0).squeeze() # -------------> culling w/ normal,  checks the z-component of the normals, if  positive, the face is facing towards the camera.
            T, C = T[NI, :], C[NI, :] # only extracts faces facing front 
            
            C = np.clip((C @ light_dir), 0, 1) # ------------------> cliping range 0 - 1, resulting in light intensity value
            C = C[:, np.newaxis].repeat(3, axis=-1) # making RGB channels
            C = np.clip(C, 0, 1) # intensity values remain within the range [0, 1]
            C = C*0.6+0.15 # reduces the intensity by 40%, ensuring brightest are not too bright / adds a small base value to the intensity, ensuring that  darkest have some brightness
            collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        else:
            C = plt.get_cmap("gray")(Z)
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]
            
            NI = np.argwhere(C[:,2] > 0).squeeze()
            T, C  = T[NI, :], C[NI, :]
            collection = PolyCollection(T, closed=False, linewidth=0.23, facecolor=C, edgecolor="black")
        
        ax.add_collection(collection)

    def update(V):
        # Cleanup previous collections
        if show_angles:
            for _ax in [ax1, ax2, ax3]:
                for coll in _ax.collections:
                    coll.remove()
            
            # Render meshes for all views
            for _ax, mvp in zip([ax1, ax2, ax3], MVPs):
                render_mesh(_ax, V, mvp, F)
                
            return ax1.collections + ax2.collections + ax3.collections
        
        else:
            for coll in single_ax.collections:
                coll.remove()
            # import pdb;pdb.set_trace()
            render_mesh(single_ax, V, MVP, F)
            
            return single_ax.collections

    ani = FuncAnimation(fig, update, frames=tqdm(Vs, desc="Rendering frames", ncols=100), blit=True) # Vs[56:202]

    ### can be saved in difference format
    os.makedirs(out_root_dir, exist_ok=True)
    video_path = f'{out_root_dir}/{video_name}.mp4'

    if render_images:
        # import pdb;pdb.set_trace()
        ani.save(video_path, writer='ffmpeg', fps=fps)
        save_frames(video_path, out_root_dir)
        
    else:
        ani.save(video_path, writer='ffmpeg', fps=fps)
    plt.close()


def render_sequence_meshes(expression_parameters_path= "", 
                           video_name="",
                           bg_black=False,
                           fps=30,
                           face_only=True,
                           remove_axis=True,
                           show_angles=True,
                           figsize=(6,6),
                           mode="mesh",
                           light_dir=np.array([0,0,1]),
                           linewidth = 1,
                           out_root_dir = "./result",
                           face_model = None,
                           ):
    
    expression_parameters = np.load(expression_parameters_path)
    if face_model == None:
        face_model = face_model_io.load_face_model('/source/inyup/TeTEC/ICT-FaceKit/FaceXModel')
    
    frame_vertices = [] # [N, 53]
    
    for frame_idx, parameter in enumerate(expression_parameters):
        # # blendshape 53
        face_model.set_expression(parameter)
        # # Deform the mesh
        face_model.deform_mesh()
        # # Write the deformed mesh
        # face_model_io.write_deformed_mesh('/data/sihun/arkit_CSH/sample_identity_arkit_frame_{:06d}.obj'.format(idx), face_model)
        # # om.write_mesh(file_path, face_model._deformed_mesh, halfedge_tex_coord = True)
        frame_vertices.append(face_model._deformed_vertices.copy())


    if bg_black:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')
    
    if show_angles:
        fig = plt.figure(figsize=(18, 6))
        ax1 = fig.add_subplot(131, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
        ax2 = fig.add_subplot(132, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
        ax3 = fig.add_subplot(133, xlim=[-1, +1], ylim=[-1, +1], aspect=1, frameon=False)
        if remove_axis:
            ax1.axis('off')
            ax2.axis('off')
            ax3.axis('off')
    else:
        fig = plt.figure(figsize=figsize)
        _r = figsize[0] / figsize[1]
        fig_xlim = [-_r,_r]
        fig_ylim = [-1,+1]
        single_ax = fig.add_axes([0,0,1,1], xlim=fig_xlim, ylim=fig_ylim, aspect=1, frameon=False)
        if remove_axis:
            single_ax.axis('off')

    if face_only:
        ## face only
        v_idx, f_idx = 9409, 9230
    else:
        ## face + head + neck
        v_idx, f_idx = 11248, 11144

    # This is quad mesh!
    quad_F  = face_model._generic_neutral_mesh.face_vertex_indices()[:f_idx] # [N,4]
    tri_faces_1 = quad_F[:, [0, 1, 2]] # [N,3]
    tri_faces_2 = quad_F[:, [0, 2, 3]] # [N,3]
    F = np.vstack([tri_faces_1, tri_faces_2]) # [2N,3]
    Vs = frame_vertices # [V,3]

    # Pre-calculated transformation matrices for three views
    if mode == "mesh":
        proj = perspective(25, 1, 1, 100) # for mesh
        model = translate(0, 0, -2.5) 
    else:
        proj = ortho(-12,12,-12,12,1,100) # for shade
        model = translate(0, 0, -2.5) 
        
    if show_angles: # if wanna render with different angles, this should be coupled with 'mesh' mode
        MVPs = [
            proj @ model,
            proj @ model @ yrotate(-30),
            proj @ model @ yrotate(-90)
        ]
    else: # if only front
        MVP = proj @ model,
        MVP = MVP[0] # IDK, but it is tupled
        # import pdb;pdb.set_trace() 
    
    def render_mesh(ax, V, MVP, F):
        # quad to triangle
        # import pdb;pdb.set_trace()
        if mode == "mesh":
            VF = transform_vertices(V[:v_idx] , MVP, F, norm=True)
        else:
            VF = transform_vertices(V[:v_idx] , MVP, F, norm=False)
        
        T = VF[:, :, :2]
        Z = -VF[:, :, 2].mean(axis=1)
        zmin, zmax = Z.min(), Z.max()
        Z = (Z - zmin) / (zmax - zmin)
        
        if mode == "shade":
            C = calc_face_norm(V, F) @ model[:3,:3].T # [3,3] -----> contains the transformed face normals
            I = np.argsort(Z) # -----------------------------------> depth sorting 
            T, C = T[I, :], C[I, :] # ensures that triangles are rendered from back to front
            
            NI = np.argwhere(C[:, 2] > 0).squeeze() # -------------> culling w/ normal,  checks the z-component of the normals, if  positive, the face is facing towards the camera.
            T, C = T[NI, :], C[NI, :] # only extracts faces facing front 
            
            C = np.clip((C @ light_dir), 0, 1) # ------------------> cliping range 0 - 1, resulting in light intensity value
            C = C[:, np.newaxis].repeat(3, axis=-1) # making RGB channels
            C = np.clip(C, 0, 1) # intensity values remain within the range [0, 1]
            C = C*0.6+0.15 # reduces the intensity by 40%, ensuring brightest are not too bright / adds a small base value to the intensity, ensuring that  darkest have some brightness
            collection = PolyCollection(T, closed=False, linewidth=linewidth, facecolor=C, edgecolor=C)
        else:
            C = plt.get_cmap("gray")(Z)
            I = np.argsort(Z)
            T, C = T[I, :], C[I, :]
            
            NI = np.argwhere(C[:,2] > 0).squeeze()
            T, C  = T[NI, :], C[NI, :]
            collection = PolyCollection(T, closed=False, linewidth=0.23, facecolor=C, edgecolor="black")
        
        ax.add_collection(collection)

    def update(V):
        # Cleanup previous collections
        if show_angles:
            for _ax in [ax1, ax2, ax3]:
                for coll in _ax.collections:
                    coll.remove()
            
            # Render meshes for all views
            for _ax, mvp in zip([ax1, ax2, ax3], MVPs):
                render_mesh(_ax, V, mvp, F)
                
            return ax1.collections + ax2.collections + ax3.collections
        
        else:
            for coll in single_ax.collections:
                coll.remove()
            # import pdb;pdb.set_trace()
            render_mesh(single_ax, V, MVP, F)
            
            return single_ax.collections

    ani = FuncAnimation(fig, update, frames=tqdm(Vs, desc="Rendering frames", ncols=100), blit=True) # Vs[56:202]

    ### can be saved in difference format
    os.makedirs(out_root_dir, exist_ok=True)
    ani.save(f'{out_root_dir}/{video_name}.mp4', writer='ffmpeg', fps=fps)
    plt.close()
    

def trim_ict_parameter(reference_video_path, expression_parameters_path, output_path, smoothing_sigmna = 1):
    
    ## to follow referene video frame number
    video = mp.VideoFileClip(reference_video_path)
    ref_num_frames = int(video.fps * video.duration)
    
    # import pdb;pdb.set_trace()
    expression_parameters = np.load(expression_parameters_path)
    
    ## apply gaussian smoothing
    # from scipy.ndimage import gaussian_filter1d
    # smoothed_parameters = gaussian_filter1d(expression_parameters, sigma=smoothing_sigmna, axis=0)
    
    ## compute delta vectors 
    tmp_expression_parameters = (expression_parameters * 1000).astype(int)
    delta_vectors = np.abs(np.diff(tmp_expression_parameters, axis=0))    
    ## sum of delta vectors 
    delta_sums = np.sum(delta_vectors, axis=1)
    second_delta_sums = np.abs(np.diff(delta_sums, axis=0))
    # third_delta_sums = np.abs(np.diff(second_delta_sums, axis=0))
    
    # change_points = np.where(third_delta_sums > np.mean(third_delta_sums))[0]
    # change_points = np.argsort(second_delta_sums)[-2:][::-1]
    change_point = second_delta_sums.argmax()
    
    # if len(change_points) == 0:
    #     print("no significant change points detected")
    #     return

    ## find the first significant change point
    end_idx = change_point + 1 
    start_idx = end_idx - ref_num_frames 
    
    new_expression_parameters = expression_parameters[start_idx : end_idx]    
    
    np.save(output_path, new_expression_parameters)
    print(f"strat index : {start_idx} / shape of trimmed expression parameter : f{new_expression_parameters.shape}")
    
    
"""
version just gives numpy array
"""  
def _trim_ict_parameter(reference_video_path, expression_parameters, output_path, smoothing_sigmna = 1):
    
    ## to follow referene video frame number
    video = mp.VideoFileClip(reference_video_path)
    ref_num_frames = int(video.fps * video.duration)
    
    ## apply gaussian smoothing
    # from scipy.ndimage import gaussian_filter1d
    # smoothed_parameters = gaussian_filter1d(expression_parameters, sigma=smoothing_sigmna, axis=0)
    
    ## compute delta vectors 
    tmp_expression_parameters = (expression_parameters * 1000).astype(int)
    delta_vectors = np.abs(np.diff(tmp_expression_parameters, axis=0))    
    ## sum of delta vectors 
    delta_sums = np.sum(delta_vectors, axis=1)
    second_delta_sums = np.abs(np.diff(delta_sums, axis=0))
    # third_delta_sums = np.abs(np.diff(second_delta_sums, axis=0))
    
    # change_points = np.where(third_delta_sums > np.mean(third_delta_sums))[0]
    # change_points = np.argsort(second_delta_sums)[-10:][::-1]
    change_point = second_delta_sums.argmax()
    
    # if len(change_points) == 0:
    #     print("no significant change points detected")
    #     return
    end_idx = change_point + 1
    start_idx = end_idx - ref_num_frames
    new_expression_parameters = expression_parameters[start_idx : end_idx]  
      
    import pdb;pdb.set_trace()
    ## usually at the last frame media player abruptly changes back to the first frame
    ## find the first significant change point
    if (start_idx) < 0: # in case the change at starting point is bigger than the change at last frame
        # import pdb;pdb.set_trace()
        end_idx = change_point + ref_num_frames
        start_idx = change_point + 1
        new_expression_parameters = expression_parameters[start_idx : end_idx + 1]    
    
    np.save(output_path, new_expression_parameters)
    
    print(f"{os.path.basename(reference_video_path)} >> strat index : {start_idx} / shape of trimmed expression parameter : f{new_expression_parameters.shape}")
    

import subprocess
def mux_audio_video(audio_path, video_path, output_path):
    cmd = f"ffmpeg -y -i {audio_path} -i {video_path} -c:v copy -map 0:a -map 1:v -c:a aac {output_path}"
    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def concat_videos(vid1_paths : list, output_path : str):
    
    # load all videos
    clips = [mp.VideoFileClip(video) for video in vid1_paths]
    
    # ensure all clips have the same height
    min_height = min(clip.h for clip in clips)
    clips = [clip.resize(height=min_height) for clip in clips]
    
    # concat videos column wise 
    final_clip = mp.clips_array([[clip] for clip in clips])
    
    final_clip.write_videofile(output_path, codec = "libx264", fps=30)
    
    return 

def bshp_2_vtx(bhsp_anim_seq : np.array = None, face_model = None):
    if face_model == None:
        face_model = face_model_io.load_face_model('/input/inyup/TeTEC/ICT-FaceKit/FaceXModel')

    vertex_animation = []
    for frame_idx, parameter in enumerate(bhsp_anim_seq):
        
        # # blendshape 53
        face_model.set_expression(parameter)
        # # Deform the mesh
        face_model.deform_mesh()
        #### TODO ####
        vs = face_model._deformed_vertices.copy() # need to flatten this and only extract full face area [0:9408]
        vertex_animation.append(vs)
    
    vertex_animation = np.array(vertex_animation)
    
    return vertex_animation 




###############
## render utils

def frustum(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = +2.0 * znear / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[0, 2] = (right + left) / (right - left)
    M[2, 1] = (top + bottom) / (top - bottom)
    M[2, 3] = -2.0 * znear * zfar / (zfar - znear)
    M[3, 2] = -1.0
    return M

def perspective(fovy, aspect, znear, zfar):
    h = np.tan(0.5*np.radians(fovy)) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)

def ortho(left, right, bottom, top, znear, zfar):
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = 2.0 / (right - left)
    M[1, 1] = 2.0 / (top - bottom)
    M[2, 2] = -2.0 / (zfar - znear)
    M[3, 3] = 1.0
    M[0, 3] = -(right + left) / (right - left)
    M[1, 3] = -(top + bottom) / (top - bottom)
    M[2, 3] = -(zfar + znear) / (zfar - znear)
    return M

def translate(x, y, z):
    return np.array([[1, 0, 0, x],
                     [0, 1, 0, y],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]], dtype=float)

def yrotate(theta):
    t = np.pi * theta / 180
    c, s = np.cos(t), np.sin(t)
    return  np.array([[ c, 0, s, 0],
                      [ 0, 1, 0, 0],
                      [-s, 0, c, 0],
                      [ 0, 0, 0, 1]], dtype=float)

def transform_vertices(frame_v, MVP, F, norm=True, no_parsing=False):
    V = frame_v
    if norm:
        V = (V - (V.max(0) + V.min(0)) / 2) / max(V.max(0) - V.min(0))
    V = np.c_[V, np.ones(len(V))]
    # import pdb;pdb.set_trace()
    V = V @ MVP.T
    V /= V[:, 3].reshape(-1, 1)
    if no_parsing:
        return V
    VF = V[F]
    return VF

def calc_face_norm(vertices, faces, mode='faces'):
    """
    Args
        vertices (np.ndarray): vertices
        faces (np.ndarray): face indices
    """

    fv = vertices[faces]
    span = fv[:, 1:, :] - fv[:, :1, :]
    norm = np.cross(span[:, 0, :], span[:, 1, :])
    norm = norm / (np.linalg.norm(norm, axis=-1)[:, np.newaxis] + 1e-12)
    
    if mode=='faces':
        return norm
    
    # Compute mean vertex normals manually
    vertex_normals = np.zeros(vertices.shape, dtype=np.float64)
    for i, face in enumerate(faces):
        for vertex in face:
            vertex_normals[vertex] += norm[i]

    # Normalize the vertex normals
    norm_v = vertex_normals / (np.linalg.norm(vertex_normals, axis=1)[:, np.newaxis] + 1e-12)

    return norm_v



if __name__ == "__main__":
    
    ########
    ## paths
    # livelink_capture_csv_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240730_MySlate_angry_3_01/MySlate_8_iPhone_raw.csv'
    # livelink_capture_csv_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240730_MySlate_neutral_1_01/MySlate_4_iPhone_raw.csv'
    # livelink_capture_csv_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240730_MySlate_sad_3_01/MySlate_9_iPhone_raw.csv'
    # livelink_capture_csv_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240731_MySlate_neutral_1_02/MySlate_10_iPhone_raw.csv'
    # livelink_capture_csv_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240731_MySlate_neutral_1_022/MySlate_11_iPhone_raw.csv'

    # output_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240730_MySlate_neutral_1_01/extracted_ict_animation.npy'
    # output_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240730_MySlate_angry_3_01/extracted_ict_animation.npy'
    # output_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240730_MySlate_sad_3_01/extracted_ict_animation.npy'
    # output_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240731_MySlate_neutral_1_02/extracted_ict_animation.npy'
    # output_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240731_MySlate_neutral_1_022/extracted_ict_animation.npy'


    ################################################################################
    ## extract expression parameters for ict morph targets from raw livelink capture
    # extract_expression_parameter(livelink_capture_csv_path, output_path)
    
    
    
    ##############################################################################
    ## trim the initial and last paused frames from extracted expression parameter
    # reference_video_path = r'/source/inyup/TeTEC/data/MEAD/ffhq_align/m03/m03-crop/m03-neutral-level_1-001.mp4' # retry 해야함(m03-crop-v1로했음 밑에 다)
    # reference_video_path = r'/source/inyup/TeTEC/data/MEAD/ffhq_align/m03/m03-crop/m03-angry-level_3-001.mp4' # retry
    # reference_video_path = r'/source/inyup/TeTEC/data/MEAD/ffhq_align/m03/m03-crop/m03-sad-level_3-001.mp4' # retry
    # reference_video_path = r'/source/inyup/TeTEC/data/MEAD/ffhq_align/m03/m03-crop/m03-neutral-level_1-002.mp4' # retry 
    # reference_video_path = r'/source/inyup/TeTEC/data/MEAD/ffhq_align/m03/m03-crop/m03-neutral-level_1-022.mp4' 
    
    # trim_output_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240730_MySlate_neutral_1_01/trim_extracted_ict_animation.npy'
    # trim_output_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240730_MySlate_angry_3_01/trim_extracted_ict_animation.npy'
    # trim_output_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240730_MySlate_sad_3_01/trim_extracted_ict_animation.npy'
    # trim_output_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240731_MySlate_neutral_1_02/trim_extracted_ict_animation.npy'
    # trim_output_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240731_MySlate_neutral_1_022/trim_extracted_ict_animation.npy'

    # trim_ict_parameter(reference_video_path, output_path, trim_output_path)
    
    
    #####################################################
    ## test rendering of dtw trimmed extracted parameters
    # trim_output_dtw_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240730_MySlate_neutral_1_01/trim_extracted_ict_animation.npy' # anchor, so there's no dtw version
    # trim_output_dtw_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240730_MySlate_neutral_1_01/trim_extracted_ict_animation_dtw.npy' # dtw version
    # trim_output_dtw_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240730_MySlate_angry_3_01/trim_extracted_ict_animation.npy' # if anchor
    # trim_output_dtw_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240730_MySlate_angry_3_01/trim_extracted_ict_animation_dtw.npy'
    # trim_output_dtw_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240730_MySlate_sad_3_01/trim_extracted_ict_animation_dtw.npy'
    # trim_output_dtw_path = r'/source/inyup/TeTEC/data/generated_data/inserted_neutral_1_01_00.npy' # keyframe inserted test
    # trim_output_dtw_path = r'/source/inyup/TeTEC/data/generated_data/inserted_neutral_1_01_angry_3_01_t1.npy' # keyframe inserted test
#     trim_output_dtw_path = r'/source/inyup/TeTEC/data/generated_data/inserted_neutral_1_01_angry_3_01_t2.npy' # keyframe inserted test
#     trim_output_dtw_path = r'/source/inyup/TeTEC/data/generated_data/inserted_neutral_1_01_angry_3_01_t3.npy' # keyframe inserted test
#     trim_output_dtw_path = r'/source/inyup/TeTEC/data/generated_data/inserted_neutral_1_01_angry_3_01_t4.npy' # keyframe inserted test
#     trim_output_dtw_path = r'/source/inyup/TeTEC/data/generated_data/inserted_neutral_1_01_angry_3_01_t5.npy' # keyframe inserted test
#     trim_output_dtw_path = r'/source/inyup/TeTEC/data/generated_data/inserted_neutral_1_01_angry_3_01_t6.npy' # keyframe inserted test

    
    # video_name = "shade_neutral_1_01"
    # video_name = "shade_neutral_1_01_dtw" # if not anchor
    # video_name = "shade_angry_3_01"
    # video_name = "shade_angry_3_01_anchor" # if anchor
    # video_name = "shade_sad_3_01_dtw" 
    # video_name = "shade_neutral_1_02"
    # video_name = "shade_neutral_1_022"
    # video_name = "shade_iserted_neutral_1_01_00"
    # video_name = "shade_iserted_neutral_1_01_angry_3_01_t1"
    # video_name = "shade_iserted_neutral_1_01_angry_3_01_t2"
#     video_name = "shade_iserted_neutral_1_01_angry_3_01_t3"
#     video_name = "shade_iserted_neutral_1_01_angry_3_01_t4"
#     video_name = "shade_iserted_neutral_1_01_angry_3_01_t5"
#     video_name = "shade_iserted_neutral_1_01_angry_3_01_t6"


    # render_sequence_meshes(trim_output_path, video_name) # extract anchor
    # render_sequence_meshes(trim_output_path, video_name, mode="shade", show_angles=False) # extract anchor, shade mode 
    # render_sequence_meshes(trim_output_dtw_path, video_name) # extract the 'dtw'ed target  
    # render_sequence_meshes(trim_output_dtw_path, video_name, mode="shade", show_angles=False) # shade mode
    # render_sequence_meshes(trim_output_dtw_path, video_name, mode="shade", show_angles=False) # shade mode
    # render_sequence_meshes(trim_output_path, video_name) # if not using dtw version
    # render_sequence_meshes(trim_output_path, video_name, mode="shade", show_angles=False) # if shaded render
    # render_sequence_meshes(trim_output_dtw_path, video_name, mode="shade", show_angles=False) # if shaded render
    
    
    # # # Video("sample_data/arkit_CSH_.mp4")
    # # # Video("sample_data/arkit_CSH.mp4")
    # Video(f"result/{video_name}.mp4")
    
    
        ##############################
        ## shaded rendering with audio

    # import pdb;pdb.set_trace()
    #############################################################
    ## test blenshape how blendshape parameters' are disentangled
    # a_trim_output_dtw_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240730_MySlate_neutral_1_01/trim_extracted_ict_animation_dtw.npy' # dtw version
    # b_trim_output_dtw_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240730_MySlate_angry_3_01/trim_extracted_ict_animation.npy' # if anchor
    # c_trim_output_dtw_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240731_MySlate_neutral_1_02/trim_extracted_ict_animation.npy'
    # c_trim_output_dtw_path = r'/source/inyup/TeTEC/data/livelink_MEAD/20240731_MySlate_neutral_1_022/trim_extracted_ict_animation.npy'
    
    # output_path = r'/source/inyup/TeTEC/data/livelink_MEAD/result/cal_parameters.npy'
    
    # a_trim_output_dtw = np.load(a_trim_output_dtw_path)
    # b_trim_output_dtw = np.load(b_trim_output_dtw_path)
    # c_trim_output_dtw = np.load(c_trim_output_dtw_path)

    # a_2_b = b_trim_output_dtw - a_trim_output_dtw
    
    # if c_trim_output_dtw.shape[0] < a_trim_output_dtw.shape[0]:
    #     match_c_trim_output = np.zeros((a_trim_output_dtw.shape[0],53)) # just clamp
    #     match_c_trim_output[:c_trim_output_dtw.shape[0]] = c_trim_output_dtw
    # else: # 다시 settting 필요
    #     match_c_trim_output = c_trim_output_dtw
           
    # cal_parameters = match_c_trim_output + a_2_b
    
    # cal_parameters = np.clip(cal_parameters, 0, 1)
    
    # # np.save(output_path, cal_parameters)
    # video_name = "check_disentanglement_t5"
    # render_sequence_meshes(output_path, video_name, mode="shade", show_angles=False)
    # Video(f"result/{video_name}.mp4")
    
    
        #######################################
        ## test only difference between a and b
    
    # expression_parameters_only = b_trim_output_dtw - a_trim_output_dtw
    #     # normalized
    # expression_parameters_only_norm = (expression_parameters_only - np.min(expression_parameters_only)) / ((expression_parameters_only - np.max(expression_parameters_only)) + 1e-8)
    # expression_parameters_only_output_path = r'/source/inyup/TeTEC/data/livelink_MEAD/result/expression_parameters_only.npy' 

    # # np.save(expression_parameters_only_output_path, expression_parameters_only)
    # np.save(expression_parameters_only_output_path, expression_parameters_only) # normalized
    
    # video_name = "shade_check_expression_parameters_only"
    # render_sequence_meshes(expression_parameters_only_output_path, video_name, mode="shade", show_angles=False)
    # Video(f"result/{video_name}.mp4")
    
    

    ######################
    ## mux audio and video
    # audio_path = "/source/inyup/TeTEC/data/MEAD/ffhq_align/m03/m03-audio/m03-neutral-level_1-002.wav"
    # audio_path = "/source/inyup/TeTEC/data/MEAD/ffhq_align/m03/m03-audio/m03-neutral-level_1-022.wav"
    # audio_path = "/source/inyup/TeTEC/data/MEAD/ffhq_align/m03/m03-audio/m03-angry-level_3-001.wav" 
    # audio_path = "/source/inyup/TeTEC/faceClip/data/MEAD/m03/m03-audio/m03-angry-level_3-003.wav" 
    # audio_path = "/source/inyup/TeTEC/faceClip/data/MEAD/m03/m03-audio/m03-angry-level_3-030.wav" 
        
    # video_path = "/source/inyup/TeTEC/data/livelink_MEAD/result/20240731_MySlate_neutral_1_02.mp4"
    # video_path = "/source/inyup/TeTEC/data/livelink_MEAD/result/check_disentanglement.mp4" # 002 content + angry expession without content
    # video_path = "/source/inyup/TeTEC/data/livelink_MEAD/result/check_disentanglement_t2.mp4" # 002 content + angry expession without content
    # video_path = "/source/inyup/TeTEC/data/livelink_MEAD/result/check_disentanglement_t3.mp4" # 022 content + angry expession without content
    # video_path = "/source/inyup/TeTEC/data/livelink_MEAD/result/check_disentanglement_t4.mp4" # 022 content + angry expession without content
    # video_path = "/source/inyup/TeTEC/data/livelink_MEAD/result/shade_concat_neut_angry_diff_noNormAdded.mp4"
    # video_path = "/source/inyup/TeTEC/data/livelink_MEAD/result/shade_iserted_neutral_1_01_angry_3_01_t2.mp4"
    # video_path = "/source/inyup/TeTEC/data/livelink_MEAD/result/20240730_MySlate_neutral_1_01_dtw.mp4" 
    # video_path = "/source/inyup/TeTEC/data/livelink_MEAD/result/20240730_MySlate_angry_3_01_anchor.mp4" # anchor
    # video_path = "/source/inyup/TeTEC/faceClip/data/result/concat_videos/cus_es_r_test_shade_ict_vtx_dtw_10000_04.mp4" 
    # video_path = "/source/inyup/TeTEC/faceClip/data/result/concat_videos/cus_eus_r_test_shade_ict_vtx_dtw_10000_04.mp4" 
    # video_path = "/source/inyup/TeTEC/faceClip/data/result/concat_videos/cs_eus_r_test_shade_ict_vtx_dtw_10000_04.mp4" 
    # video_path = "/source/inyup/TeTEC/faceClip/data/result/concat_videos/cs_es_r_test_shade_ict_vtx_dtw_10000_04.mp4" 

    # result_path = "/source/inyup/TeTEC/data/livelink_MEAD/result/20240731_MySlate_neutral_1_02_dtw_mux.mp4"
    # result_path = "/source/inyup/TeTEC/data/livelink_MEAD/result/check_disentanglement_mux.mp4"
    # result_path = "/source/inyup/TeTEC/data/livelink_MEAD/result/check_disentanglement_t2_mux.mp4"
    # result_path = "/source/inyup/TeTEC/data/livelink_MEAD/result/check_disentanglement_t3_mux.mp4"
    # result_path = "/source/inyup/TeTEC/data/livelink_MEAD/result/check_disentanglement_t4_mux.mp4"
    # result_path = "/source/inyup/TeTEC/data/livelink_MEAD/result/20240730_MySlate_neutral_1_01_dtw_mux.mp4"
    # result_path = "/source/inyup/TeTEC/data/livelink_MEAD/result/20240730_MySlate_angry_3_01_anchor_mux.mp4"
    # result_path = "/source/inyup/TeTEC/data/livelink_MEAD/result/shade_concat_neut_angry_diff_mux.mp4"
    # result_path = "/source/inyup/TeTEC/data/livelink_MEAD/result/shade_concat_neut_angry_diff_noNormAdded_mux.mp4"
    # result_path = "/source/inyup/TeTEC/data/livelink_MEAD/result/shade_iserted_neutral_1_01_angry_3_01_t2_mux.mp4"
    # result_path = "/source/inyup/TeTEC/faceClip/data/result/concat_videos/cus_es_r_test_shade_ict_vtx_dtw_10000_04_mux.mp4" 
    # result_path = "/source/inyup/TeTEC/faceClip/data/result/concat_videos/cus_eus_r_test_shade_ict_vtx_dtw_10000_04_mux.mp4" 
    # result_path = "/source/inyup/TeTEC/faceClip/data/result/concat_videos/cs_eus_r_test_shade_ict_vtx_dtw_10000_04_mux.mp4"
    # result_path = "/source/inyup/TeTEC/faceClip/data/result/concat_videos/cs_es_r_test_shade_ict_vtx_dtw_10000_04_mux.mp4"

    
    # mux_audio_video(audio_path, video_path, result_path)
    
    ################
    # temporary test 
    # import pickle
    # import pdb;pdb.set_trace()
    # path = '/source/inyup/TeTEC/data/MEAD/ffhq_align/m03/dataset_m003_vtx_dtw.pickle'
    # with open(path, 'rb') as f:
    #     data = pickle.load(f)
    
    
    ################
    ## concat videos
    
    # vid_paths = [ 
                #  "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/12/angry_3_003/a_shade_lipvtx_trim_extracted_ict_animation_dtw.mp4",
                #  "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/12/happy_3_003/a_shade_lipvtx_trim_extracted_ict_animation_dtw.mp4",
                #  "/source/inyup/TeTEC/faceClip/data/test/result/qualitative_render/IEFA/angry_003_happy_003_disict_vtx_dtw_10000_03_mux.mp4",
                
                # "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/12/angry_3_003/a_shade_lipvtx_trim_extracted_ict_animation_dtw.mp4",
                # "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/11/happy_3_030/a_shade_lipvtx_trim_extracted_ict_animation_dtw.mp4",
                # "/source/inyup/TeTEC/faceClip/data/test/result/qualitative_render/IEFA/angry_003_happy_030_disict_vtx_dtw_10000_03_mux.mp4",
                
                # "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/11/angry_3_030/a_shade_lipvtx_trim_extracted_ict_animation_dtw.mp4",
                # "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/11/happy_3_030/a_shade_lipvtx_trim_extracted_ict_animation_dtw.mp4",
                # "/source/inyup/TeTEC/faceClip/data/test/result/qualitative_render/IEFA/angry_030_happy_030_disict_vtx_dtw_10000_03_mux.mp4",
                
                # "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/11/angry_3_030/a_shade_lipvtx_trim_extracted_ict_animation_dtw.mp4",
                # "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/12/happy_3_003/a_shade_lipvtx_trim_extracted_ict_animation_dtw.mp4",
                # "/source/inyup/TeTEC/faceClip/data/test/result/qualitative_render/IEFA/angry_030_happy_003_disict_vtx_dtw_10000_03_mux.mp4",
                
                
                 
                #  "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/12/angry_3_003/s_shade_lipvtx_trim_extracted_ict_animation_dtw.mp4",
                #  "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/12/happy_3_003/s_shade_lipvtx_trim_extracted_ict_animation_dtw.mp4",
                #  "/source/inyup/TeTEC/faceClip/data/test/result/qualitative_render/IEFA/angry_003_happy_003_disict_vtx_dtw_10000_04_mux.mp4",
                
                # "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/12/angry_3_003/s_shade_lipvtx_trim_extracted_ict_animation_dtw.mp4",
                # "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/11/happy_3_030/s_shade_lipvtx_trim_extracted_ict_animation_dtw.mp4",
                # "/source/inyup/TeTEC/faceClip/data/test/result/qualitative_render/IEFA/angry_003_happy_030_disict_vtx_dtw_10000_04_mux.mp4",
                
                # "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/11/angry_3_030/s_shade_lipvtx_trim_extracted_ict_animation_dtw.mp4",
                # "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/11/happy_3_030/s_shade_lipvtx_trim_extracted_ict_animation_dtw.mp4",
                # "/source/inyup/TeTEC/faceClip/data/test/result/qualitative_render/IEFA/angry_030_happy_030_disict_vtx_dtw_10000_04_mux.mp4",
                
                # "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/11/angry_3_030/s_shade_lipvtx_trim_extracted_ict_animation_dtw.mp4",
                # "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/12/happy_3_003/s_shade_lipvtx_trim_extracted_ict_animation_dtw.mp4",
                # "/source/inyup/TeTEC/faceClip/data/test/result/qualitative_render/IEFA/angry_030_happy_003_disict_vtx_dtw_10000_04_mux.mp4",
                 
                # "/source/inyup/TeTEC/data/livelink_MEAD/result/shade_neutral_1_01_dtw.mp4",
                #  "/source/inyup/TeTEC/data/livelink_MEAD/result/shade_angry_3_01_anchor.mp4",
                #  "/source/inyup/TeTEC/data/livelink_MEAD/result/shade_check_expression_parameters_only_noNorm.mp4",
                #  "/source/inyup/TeTEC/data/livelink_MEAD/result/shade_check_expression_parameters_only.mp4",
                
                # ]
                
    # output_path = "/source/inyup/TeTEC/faceClip/data/result/concat_videos/cus_eus_r_test_shade_ict_vtx_dtw_10000_03.mp4"
    # output_path = "/source/inyup/TeTEC/faceClip/data/result/concat_videos/cus_es_r_test_shade_ict_vtx_dtw_10000_03.mp4" 
    # output_path = "/source/inyup/TeTEC/faceClip/data/result/concat_videos/cs_es_r_test_shade_ict_vtx_dtw_10000_03.mp4"    
    # output_path = "/source/inyup/TeTEC/faceClip/data/result/concat_videos/cs_eus_r_test_shade_ict_vtx_dtw_10000_03.mp4"
    
    # output_path = "/source/inyup/TeTEC/faceClip/data/result/concat_videos/cus_eus_r_test_shade_ict_vtx_dtw_10000_04.mp4"
    # output_path = "/source/inyup/TeTEC/faceClip/data/result/concat_videos/cus_es_r_test_shade_ict_vtx_dtw_10000_04.mp4" 
    # output_path = "/source/inyup/TeTEC/faceClip/data/result/concat_videos/cs_es_r_test_shade_ict_vtx_dtw_10000_04.mp4"    
    # output_path = "/source/inyup/TeTEC/faceClip/data/result/concat_videos/cs_eus_r_test_shade_ict_vtx_dtw_10000_04.mp4"
    
    # concat_videos(vid_paths, output_path)
    
    
    ##########################
    ## rename capture folders
    # import glob
    # dir = "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/0"
    # dir = "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/1"
    # dir = "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/2"

    # folders = sorted([f for f in glob.glob(os.path.join(dir, "*")) if os.path.isdir(f)])
#     emo_list = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised'] # 
#     sentence_idx = int(dir.split('/')[-1])
    
#     sentence_M003 = [
#                 ['001','001','028','001','001','001','001','001'], # Sentence A
#                 ['002','002','029','002','002','002','002','002'], # Sentence B
#                 ['021','020','018','021','021','031','021','021'], # ...
#                 ['022','021','019','022','022','032','022','022'],
#                 ['023','022','020','023','023','033','023','023'],
#                 ['024','023','021','024','024','034','024','024'],
#                 ['025','024','022','025','025','035','025','025'],
#                 ['026','025','023','026','026','036','026','026'],
#                 ['027','026','024','027','027','037','027','027'],
#                 ['028','027','025','028','028','038','028','028'],
#                 ['029','028','026','029','029','039','029','029'],
#                 ['030','029','027','030','030','040','030','030'],
#                 ['003','003','030','003','003','003','003','003']
#             ]
    
#     for i, old_path in enumerate(folders):
#         old_name = os.path.basename(old_path)
#         if emo_list[i] == 'neutral':
#             level = 1
#         else:
#             level = 3
#         new_name = f"{emo_list[i]}_{level}_{sentence_M003[sentence_idx][i]}"
#         new_path = os.path.join(dir, new_name)
        
#         if os.path.exists(old_path) and os.path.isdir(old_path):
#             os.rename(old_path, new_path)


    ###############################
    ## deleting 'Myslate" at front
    # for i, old_path in enumerate(folders):
    #     # import pdb;pdb.set_trace()
    #     old_name = os.path.basename(old_path)
        
    #     new_name = old_name.replace('Myslate_', "")
    #     new_path = os.path.join(dir, new_name)
        
    #     if os.path.exists(old_path) and os.path.isdir(old_path):
    #         os.rename(old_path, new_path)
    
    
    #####################################################################################################
    ## converting ICT captured data(bshp param) into vtx coordinates as FaceCLIP's emoca dataset(.pickle)
    
    import pickle
    import glob
    
    # data_root_path = "/source/inyup/TeTEC/faceClip/data/livelink_MEAD"
    # data_output_path = "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/ict_dataset_m003_vtx_dtw_nolevel.pickle" # set output
    # data_output_path = "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/ict_dataset_m003_vtx_dtw_nolevel_woneut_03.pickle" # anchor is shortest seq, lip vertex dtw version
    # data_output_path = "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/ict_dataset_m003_vtx_dtw_nolevel_woneut_04.pickle" # anchor is angry, lip vertex dtw version
        
    # sentence_list = ["0","1","2","3","4","5","6","7","8","9","10","11","12"]
    # bshp = False
    # # sentences_folders = sorted([d for d in glob.glob(os.path.join(data_root_path, "*")) if os.path.isdir(d)])
    # sentences_folders = []
    # for s in sentence_list:
    #     sentences_folders.append(os.path.join(data_root_path, s))
    # result_dic = {}
    # face_model = face_model_io.load_face_model('/source/inyup/TeTEC/ICT-FaceKit/FaceXModel')
    # for id, sentences_folder in enumerate(sentences_folders):
    #     ## insdie the loop, start for loop iterate over all (A, B) occurrences for M emotions (ex. if [a,b,c,d,e,f,g,h] -> (a,a), (a,b) ... (a,h), (b,a), (b,b)...(h,g),(h,h)), over the folders(emotions, M number of them) inside the folders(sentences)
    #     emotion_folders = sorted([d for d in glob.glob(os.path.join(sentences_folder, "*"))])
    #     sentence_result_dic = {}
    #     for emotion_folder in emotion_folders:
    #         # import pdb;pdb.set_trace()
    #         base_name = os.path.basename(emotion_folder)
    #         emotion_name = base_name.split('_')[0]
    #         if emotion_name == 'neutral': # skip if neutral
    #             continue
    #         sentence_name = base_name.split('_')[-1]
    #         animation_parameter_key = f"{emotion_name}_3_M003_front_{emotion_name}_3_{sentence_name}"
    #         # animation_parameter_path = os.path.join(emotion_folder, "trim_extracted_ict_animation_dtw.npy")
    #         animation_parameter_path = os.path.join(emotion_folder, "s_lipvtx_trim_extracted_ict_animation_dtw.npy") # lip vertex dtw version but anchor is shortest sequence
    #         # animation_parameter_path = os.path.join(emotion_folder, "a_lipvtx_trim_extracted_ict_animation_dtw.npy") # lip vertex dtw version but anchor is angry

    #         animation_parameter = np.load(animation_parameter_path)
    #         vertex_animation = []
    #         for frame_idx, parameter in enumerate(animation_parameter):
    #             if bshp:
    #                 # # blendshape 53
    #                 face_model.set_expression(parameter)
    #                 # # Deform the mesh
    #                 face_model.deform_mesh()
    #                 #### TODO ####
    #                 vs = face_model._deformed_vertices.copy() # need to flatten this and only extract full face area [0:9408]
    #             else:
    #                 vertex_animation.append(parameter)
    #                 # import pdb;pdb.set_trace()
            
    #         vertex_animation = np.array(vertex_animation) 
    #         sentence_result_dic[animation_parameter_key] = vertex_animation # save it here
        
    #     result_dic[str(id)] = sentence_result_dic
    
    ## save result_dic 
    # with open(data_output_path, "wb") as f:
    #     pickle.dump(result_dic, f) 
    
    
    # ## saved data sanity check
    # dtw_path = "/source/inyup/TeTEC/faceClip/data/old/feature/dataset_m003_vtx_dtw.pickle"
    # dtw_path = "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/ict_dataset_m003_vtx_dtw_nolevel.pickle"
    # dtw_path = "/source/inyup/TeTEC/faceClip/data/feature/ict_dataset_m003_vtx_dtw_nolevel_woneut_04.pickle" # shortest seq anchor 
    # dtw_path = "/source/inyup/TeTEC/faceClip/data/feature/ict_dataset_m003_vtx_dtw_nolevel_woneut_03.pickle" # angry anchor 

    # with open(dtw_path, "rb") as f:
    #     dtw_data = pickle.load(f)
    # import pdb;pdb.set_trace()

    # expression_parameter = dtw_data["12"]["angry_3_M003_front_angry_3_003"]
    # video_name = "angry_3_M003_front_angry_3_003"
    # output_path = "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/12/angry_3_003"
    # _render_sequence_meshes(expression_parameters=expression_parameter, video_name=video_name, mode="shade", show_angles=False, out_root_dir=output_path)
    # Video(f"{output_path}/{video_name}.mp4")
    
    
    ##############################################################
    # make neutral render video to look for neutral face parameter
    # import pdb;pdb.set_trace()
    # face_model = face_model_io.load_face_model('/source/inyup/TeTEC/ICT-FaceKit/FaceXModel')
    
    # ## render neutral as well 
    # neutral_parameters_path_list = ['/source/inyup/TeTEC/faceClip/data/livelink_MEAD/0/neutral_1_001',
    #                       '/source/inyup/TeTEC/faceClip/data/livelink_MEAD/1/neutral_1_002',
    #                       ]
    
    # video_name = "shade_trim_extracted_ict_animation_dtw"
    # for neutral_parameters_path in neutral_parameters_path_list:
    #     expression_parameters = np.load(os.path.join(neutral_parameters_path, "trim_extracted_ict_animation_dtw.npy"))
    #     _render_sequence_meshes(expression_parameters=expression_parameters, video_name=video_name, mode="shade", show_angles=False, out_root_dir=neutral_parameters_path, face_model=face_model)
    #     Video(f"{neutral_parameters_path}/{video_name}.mp4")
    
    ####################################
    ## save neutral ict_m003
    # face_model = face_model_io.load_face_model('/input/inyup/TeTEC/ICT-FaceKit/FaceXModel')
    # paramters = np.load('/input/inyup/TeTEC/faceClip/data/livelink_MEAD/0/neutral_1_001/trim_extracted_ict_animation_dtw.npy')
    # parameter = paramters[-1,:] # retrieve the last one
    # import pdb;pdb.set_trace()
    # output_path = ""
    # face_model.set_expression(parameter)
    # # # Deform the mesh
    # face_model.deform_mesh()
    # #### TODO ####
    # vs = face_model._deformed_vertices.copy() # need to flatten this and only extract full face area [0:9408]
    # vs = vs[0:9409,:].flatten()
    # vs = np.array(vs)
    # np.save("/source/inyup/TeTEC/faceClip/data/livelink_MEAD/ict_M003_front_neutral_1_011_last_fr.npy", vs) # neutral parameter는 0 벡터가 아님
    
    #############################################
    ## render with vertices output from 1st stage
    # pred_VS_path = "/source/inyup/TeTEC/faceClip/data/test/result/vtx/m03-angry-level_3-003_disict_default.npy"
    # pred_VS_path = "/source/inyup/TeTEC/faceClip/data/test/result/vtx/m03-angry-level_3-003_disict_600.npy"
    # pred_VS_path = "/source/inyup/TeTEC/faceClip/data/test/result/vtx/m03-angry-level_3-003_disict_1200.npy"
    # pred_VS_path = "/source/inyup/TeTEC/faceClip/data/test/result/vtx/m03-angry-level_3-003_disict_10000.npy"
    # pred_VS_path = "/source/inyup/TeTEC/faceClip/data/test/result/vtx/angry_003_happy_022_disict_10000.npy"
    # pred_VS_path = "/source/inyup/TeTEC/faceClip/data/test/result/vtx/angry_003_happy_021_disict_10000.npy"

    # # Vs = np.load(pred_VS_path)
    # # Vs = Vs.reshape(45,-1,3)
    
    # save_dir = "/source/inyup/TeTEC/faceClip/data/test/result/qualitative_render/IEFA"
    # video_name = "con_angry_003_exp_happy_021-600"
    # video_name = "con_angry_003_exp_happy_021-1200"
    # video_name = "con_angry_003_exp_happy_021-10000"
    # video_name = "con_angry_003_exp_angry_003-10000"
    # video_name = "con_angry_003_exp_happy_022-10000"
    # video_name = "con_angry_003_exp_happy_021-10000"
    # audio_path = "/source/inyup/TeTEC/faceClip/data/test/audio/userstudy/m03-angry-level_3-003.wav"

    # video_path = os.path.join(save_dir, video_name+".mp4")
    # final_mux_result_path = video_path.replace('.mp4', '_mux.mp4')
    
    # v_render_sequence_meshes(
    #     Vs_path=pred_VS_path,
    #     video_name=video_name,
    #     mode='shade',
    #     out_root_dir = save_dir,
    #     face_model=face_model,
    # )
    # mux_audio_video(audio_path, video_path, final_mux_result_path)
    
    #################################################
    ## render images with custom expression paramters
    
    # expression_paramters = np.identity(53)
    # expression_paramters[0][0] = 0
    # _render_sequence_meshes(expression_parameters=expression_paramters,
    #                         render_images=True)
    
    
    ###################
    ## param check
    # orig_param_path = "/source/inyup/TeTEC/faceClip/data/livelink_MEAD/0/happy_3_001_wrong/trim_extracted_ict_animation.npy"
    # orig_param = np.load(orig_param_path)
    # """
    # 10 
    #     - contempt : 128, 53
    #     - disgusted : 102, 53
    #     - happy : 111, 53
    #     - neutral : 99, 53
    #     - fear : 87,53          x
    #     - angry : 95, 53        x
    #     - surprised : 96, 53    x
    #     - sad : 87, 53          x
    # 0 
    #     - contempt : 96,53      x
    #     - disgusted : 123, 53
    #     - happy : 96, 53        x (very wrong)
    #     - neutral : 111, 53
    #     - fear : 84, 53         x
    #     - angry : 95, 53        x
    #     - surprised : 113, 53
    #     - sad : 105, 53         x
        
    #     >> anchor seq(=neutral) 보다 frame 수가 적으면 popping 현상이 더 일어나는 현상이 있다  
    # """    
    # import pdb;pdb.set_trace()