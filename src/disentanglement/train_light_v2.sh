## does the same thing with train.py but lighter for memory usage. 

## v6 (WIP) : only decoder training and with soft lip loss to all face mesh (not only lip vertices) but with epsilon 3.0 -> 3.5 (WIP)
python train_light_v2.py --model_num "BEST_ict_vtx_v6" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## v7 (WIP): only decoder training and with soft lip loss to all face mesh (not only lip vertices) but with epsilon 3.5 -> 3.75 (WIP)
python train_light_v2.py --model_num "BEST_ict_vtx_v7" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## v8 (WIP): v7에서 training more with emotion 
python train_light_v2.py --model_num "BEST_ict_vtx_v7" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## v9 epsilon 3.5 (scale matched) 
python train_light_v2.py --w_cross 0.5 --w_self 0.5 --w_con 0.01 --w_tpl 0.01 --w_lip_strong 0.3 --w_lip_loose 4.0 --model_num "BEST_ict_vtx_v9" --use_lip_contact_loss True --use_soft_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## v10 epsilon 4.0 (scale matched) 
python train_light_v2.py --w_cross 0.5 --w_self 0.5 --w_con 0.01 --w_tpl 0.01 --w_lip_strong 0.3 --w_lip_loose 4.0 --model_num "BEST_ict_vtx_v10" --use_lip_contact_loss True --use_soft_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"
