## does the same thing with train.py but lighter for memory usage. 
# ## v1 (WIP) : only decoder training and with soft lip loss to all face mesh (not only lip vertices) but with epsilon 3.0 -> 3.5 (
# python train_light_v2_dyn.py --model_num "ict_vtx_diffrom_dyn_v1" --use_dynamic True --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True 
# python train_light_v2_dyn.py --model_num "ict_vtx_diffrom_dyn_v1_facs_dynamic_e200" --use_dynamic True --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True 

# python train_light_v2_dyn.py --model_num "ict_vtx_diffrom_dyn_v1_1" --use_train_con True --use_dynamic True --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True 

# ## v1 - 1 (WIP) : only decoder training and with soft lip loss to all face mesh (not only lip vertices) but with 2x epochs 
# python train_light_v2_dyn.py --bshp_epochs 4000 --facs_epochs 400 --emot_epochs 100 --model_num "ict_vtx_diffrom_dyn_v2" --use_dynamic True --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True 
# python train_light_v2_dyn.py --bshp_epochs 4000 --facs_epochs 400 --emot_epochs 100 --model_num "ict_vtx_diffrom_dyn_v2_facs_dynamic_e400" --use_dynamic True --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True 

# python train_light_v2_dyn.py --use_train_con True --bshp_epochs 4000 --facs_epochs 400 --emot_epochs 100 --model_num "ict_vtx_diffrom_dyn_v2_1" --use_dynamic True --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True 

# ## v1 - 2 (WIP) : only decoder training and with soft lip loss to all face mesh (not only lip vertices) but with 2x epochs 
# python train_light_v2_dyn.py --bshp_epochs 3000 --facs_epochs 300 --emot_epochs 75 --model_num "ict_vtx_diffrom_dyn_v3" --use_dynamic True --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True 
# python train_light_v2_dyn.py --bshp_epochs 3000 --facs_epochs 300 --emot_epochs 75 --model_num "ict_vtx_diffrom_dyn_v3_facs_static_e200" --use_dynamic True --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True 

# python train_light_v2_dyn.py --use_train_con True --bshp_epochs 3000 --facs_epochs 300 --emot_epochs 75 --model_num "ict_vtx_diffrom_dyn_v3_1" --use_dynamic True --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True 


## v2 (WIP): only decoder training and with soft lip loss to all face mesh (not only lip vertices) but with epsilon 3.5 -> 3.75 
# python train_light_v2_dyn.py --model_num "ict_vtx_diffrom_dyn_v2" --use_dynamic True --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True 


## v3 (WIP): v7에서 training more with emotion  
# python train_light_v2_dyn.py --model_num "ict_vtx_diffrom_dyn_v3" --use_dynamic True --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True 


## v4 epsilon 3.5 (scale matched) 
# python train_light_v2_dyn.py --model_num "BEST_ict_vtx_v9" --w_cross 0.5 --w_self 0.5 --w_con 0.01 --w_tpl 0.01 --w_lip_strong 0.3 --w_lip_loose 4.0 --use_lip_contact_loss True --use_soft_lip_contact_loss True --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"


# ## v5 epsilon 4.0 (scale matched) 
# python train_light_v2_dyn.py --model_num "BEST_ict_vtx_v10" --w_cross 0.5 --w_self 0.5 --w_con 0.01 --w_tpl 0.01 --w_lip_strong 0.3 --w_lip_loose 4.0 --use_lip_contact_loss True --use_soft_lip_contact_loss True --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"


###############
## SIGA 2025 
###############

# ## v1 (WIP) : only decoder training and with soft lip loss to all face mesh (not only lip vertices) but with epsilon 3.0 -> 3.5 (
# python train_light_v2_dyn.py --model_num "ict_vtx_diffrom_dyn_v1_facs_dynamic_e200" --use_dynamic True --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True 
# ## v1 (WIP) : with soft lip loss to all face mesh (not only lip vertices) but with epsilon 3.0 -> 3.5 (
# python train_light_v2_dyn.py --model_num "ict_vtx_diffrom_dyn_v1_1" --use_train_con True --use_dynamic True --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True 

# ## v1 - 1 (WIP) : only decoder training and with soft lip loss to all face mesh (not only lip vertices) but with 2x epochs 
# python train_light_v2_dyn.py --bshp_epochs 4000 --facs_epochs 400 --emot_epochs 100 --model_num "ict_vtx_diffrom_dyn_v2_facs_dynamic_e400" --use_dynamic True --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True 
# ## v1 - 1 (WIP) : with soft lip loss to all face mesh (not only lip vertices) but with 2x epochs 
# python train_light_v2_dyn.py --use_train_con True --bshp_epochs 4000 --facs_epochs 400 --emot_epochs 100 --model_num "ict_vtx_diffrom_dyn_v2_1" --use_dynamic True --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True 

# ## v1 - 2 (WIP) : only decoder training and with soft lip loss to all face mesh (not only lip vertices) but with 2x epochs 
# ## v1 - 2 (WIP) : with soft lip loss to all face mesh (not only lip vertices) but with 2x epochs 
# python train_light_v2_dyn.py --bshp_epochs 3000 --facs_epochs 300 --emot_epochs 75 --model_num "ict_vtx_diffrom_dyn_v3_facs_static_e200" --use_dynamic True --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True 
# python train_light_v2_dyn.py --use_train_con True --bshp_epochs 3000 --facs_epochs 300 --emot_epochs 75 --model_num "ict_vtx_diffrom_dyn_v3_1" --use_dynamic True --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True 

## v2 : with soft lip loss to all face mesh (not only lip vertices) but with epsilon 3.0 -> 3.5 
## with stage 1-3 (in ROM magnitude) , only static (not using dynamic)
python train_light_v2_dyn.py --model_num "ict_vtx_diffrom_stage_v1" --use_train_con True --use_curriculum_training True --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 
python train_light_v2_dyn.py --model_num "ict_vtx_diffrom_stage_v1_s1_final" --use_train_con True --use_curriculum_training True --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 
