"""
###########
### log ###
###########

## 2024-08-21
modified version of test.py of original FaceCLIP 
: To render ICT-FaceKit mesh
## 2024-09-02
added direct_decoding() for use in run_IEFA.py
 
"""

import sys
sys.path.insert(0, '../')
import os
import torch
import pickle
import numpy as np
from datetime import datetime
from parser_util import disentangle_args
from disentanglement.train import Runner
from render import render_from_vertex
sys.path.append('./data/livelink_MEAD')
from util import v_render_sequence_meshes, mux_audio_video

## TODO 
## - (DONE) regard this code and parser_util.py to configure 'add_IEFA_options' function
def direct_decoding(hparams, motion_num, con_vtx_anim_seq, exp_vtx_anim_seq, audio_path = None, runner= None, render = True):
    with torch.no_grad():
        device = torch.device("cuda:" + str(hparams.device - 1)) if hparams.device > 0 else torch.device("cpu")
        filename = datetime.now().date().strftime("%Y%m%d") + "_" + str(motion_num)
        
        os.makedirs(f'{hparams.root_dir + hparams.save_vtx_dir}', exist_ok=True)
        os.makedirs(f'{hparams.root_dir + hparams.save_video_dir}', exist_ok=True)
        
        vtx_path = f'{hparams.root_dir + hparams.save_vtx_dir}/{filename}_dis_{hparams.model_num}.npy'
        vid_path = f'{hparams.root_dir + hparams.save_video_dir}/{filename}_dis_{hparams.model_num}.mp4'
        
        vtx_con = torch.from_numpy(con_vtx_anim_seq).to(device).float()
        vtx_exp = torch.from_numpy(exp_vtx_anim_seq).to(device).float()
        pred_vtx = runner.autoencoder.reconstruct([vtx_con, vtx_exp])

        pred_vtx = pred_vtx.squeeze()
        pred = pred_vtx.cpu().numpy()

        np.save(vtx_path, pred) 
        
        # save video
        if render:
            video_name = os.path.basename(vid_path).split('.')[0]
            final_mux_result_path = vid_path.replace('.mp4','_mux.mp4')
            v_render_sequence_meshes(Vs_path = vtx_path,
                                     video_name=video_name,
                                     mode='shade',
                                     out_root_dir=hparams.root_dir + hparams.save_video_dir)
            if audio_path != None:
                mux_audio_video(audio_path, vid_path, final_mux_result_path)
                os.remove(vid_path)
                print("[{}] mux video vtx file saved. Frame length: {}".format(filename, pred.shape[0]))
            else:
                print("[{}] video and vtx file saved. Frame length: {}".format(filename, pred.shape[0]))

        return pred # outputs numpy array   


def test(hparams, runner, vtx_dict, audio_path):
    with torch.no_grad():
        device = torch.device("cuda:" + str(hparams.device - 1)) if hparams.device > 0 else torch.device("cpu")
        # filename = os.path.splitext(hparams.test_audio)[0]
        filename = vtx_dict['filename']
        
        os.makedirs(f'{hparams.root_dir + hparams.save_vtx_dir}', exist_ok=True)
        os.makedirs(f'{hparams.root_dir + hparams.save_video_dir}', exist_ok=True)
        
        vtx_path = f'{hparams.root_dir + hparams.save_vtx_dir}/{filename}_dis{hparams.model_num}.npy'
        vid_path = f'{hparams.root_dir + hparams.save_video_dir}/{filename}_dis{hparams.model_num}.mp4'
        # import pdb;pdb.set_trace()
        vtx_con = torch.from_numpy(vtx_dict['con']).to(device).float()
        vtx_exp = torch.from_numpy(vtx_dict['exp']).to(device).float()
        pred_vtx = runner.autoencoder.reconstruct([vtx_con, vtx_exp])

        pred_vtx = pred_vtx.squeeze()
        pred = pred_vtx.cpu().numpy()

        # save video
        np.save(vtx_path, pred)
        # render_from_vertex(hparams, vtx_path, audio_path, vid_path)
        video_name = os.path.basename(vid_path).split('.')[0]
        final_mux_result_path = vid_path.replace('.mp4','_mux.mp4')
        v_render_sequence_meshes(Vs_path = vtx_path,video_name=video_name,mode='shade', out_root_dir=hparams.root_dir + hparams.save_video_dir)
        mux_audio_video(audio_path, vid_path, final_mux_result_path)
        os.remove(vid_path)
        print("[{}] Saved. Frame length: {}".format(filename, pred.shape[0]))


if __name__ == '__main__':
    hparams = disentangle_args()
    device = torch.device("cuda:" + str(hparams.device - 1)) if hparams.device > 0 else torch.device("cpu")

    print(" ------ Test Model : {} ------- ".format(hparams.model_num))
    model_path = f"{os.path.join(hparams.root_dir , hparams.save_dir)}/{hparams.model_num}.pth"
    runner = Runner(hparams)
    # import pdb;pdb.set_trace()
    runner.autoencoder.load_state_dict(torch.load(model_path, map_location='cuda:0')['autoencoder_state_dict'])
    runner.autoencoder.eval()

    # Test file
    f = open(hparams.root_dir + hparams.vtx_dtw_path, 'rb')
    vtx = pickle.load(f)
    f.close()

    ##############################
    ## different con different exp
    content, emotion = 'angry', 'happy'
    vtx_dict = {}
    # vtx_dict['con'] = np.load(hparams.root_dir + hparams.test_con_vtx)
    # vtx_dict['exp'] = np.load(hparams.root_dir + hparams.test_exp_vtx)

    ## cus_eus
    # con_list = [content, '12', '003']
    # exp_list = [emotion, '12', '003']
    # ## cus_es
    # con_list = [content, '12', '003']
    # exp_list = [emotion, '11', '030']
    ## cs_es
    # con_list = [content, '11', '030']
    # exp_list = [emotion, '11', '030']
    ## cs_eus
    con_list = [content, '11', '030']
    exp_list = [emotion, '12', '003']
    
    filename = f'{content}_{con_list[-1]}_{emotion}_{exp_list[-1]}'    # import pdb;pdb.set_trace()
    
    vtx_dict['con'] = vtx[f'{con_list[1]}'][f'{content}_3_M003_front_{content}_3_{con_list[-1]}']
    vtx_dict['exp'] = vtx[f'{exp_list[1]}'][f'{emotion}_3_M003_front_{emotion}_3_{exp_list[-1]}']
    
    ## if any seqence longer than other, just clip it
    if vtx_dict['con'].shape[0] < vtx_dict['exp'].shape[0]:
        vtx_dict['exp'] = vtx_dict['exp'][:vtx_dict['con'].shape[0]]
    else:
        vtx_dict['con'] = vtx_dict['con'][:vtx_dict['exp'].shape[0]]
        
    vtx_dict['filename'] = '{}'.format(filename)
    test_filepath = hparams.root_dir + hparams.test_wav_dir +'/'+ hparams.test_audio
    test(hparams, runner, vtx_dict, test_filepath)
    
    # ##########################
    # ## same con different exp
    # content, emotion = 'angry', 'happy'
    # vtx_dict = {}
    # # vtx_dict['con'] = np.load(hparams.root_dir + hparams.test_con_vtx)
    # # vtx_dict['exp'] = np.load(hparams.root_dir + hparams.test_exp_vtx)

    # con_list = [content, '3', '003']
    # exp_list = [emotion, '3', '021']
    # filename = f'{content}_{con_list[-1]}_{emotion}_{exp_list[-1]}'    # import pdb;pdb.set_trace()
    # # import pdb;pdb.set_trace()
    
    # vtx_dict['con'] = vtx['4'][f'{content}_{con_list[1]}_M003_front_{content}_3_{con_list[-1]}']
    # vtx_dict['exp'] = vtx['5'][f'{emotion}_{exp_list[1]}_M003_front_{emotion}_3_{exp_list[-1]}'][:vtx_dict['con'].shape[0]]
    # vtx_dict['filename'] = '{}'.format(filename)
    # test_filepath = hparams.root_dir + hparams.test_wav_dir +'/'+ hparams.test_audio
    # test(hparams, runner, vtx_dict, test_filepath)
    
    #################################
    ## same con same exp (self recon)
    # content, emotion = 'angry', 'angry'
    # vtx_dict = {}
    # # vtx_dict['con'] = np.load(hparams.root_dir + hparams.test_con_vtx)
    # # vtx_dict['exp'] = np.load(hparams.root_dir + hparams.test_exp_vtx)

    # con_list = [content, '3', '003']
    # exp_list = [emotion, '3', '003']
    # filename = f'{content}_{con_list[-1]}_{emotion}_{exp_list[-1]}'
    # # import pdb;pdb.set_trace()
    
    # vtx_dict['con'] = vtx['4'][f'{content}_{con_list[1]}_M003_front_{content}_3_{con_list[-1]}']
    # vtx_dict['exp'] = vtx['4'][f'{emotion}_{exp_list[1]}_M003_front_{emotion}_3_{exp_list[-1]}'][:vtx_dict['con'].shape[0]]
    # vtx_dict['filename'] = '{}'.format(filename)
    # test_filepath = hparams.root_dir + hparams.test_wav_dir +'/'+ hparams.test_audio
    # test(hparams, runner, vtx_dict, test_filepath)
