## does the same thing with train.py but lighter for memory usage. 

## v6 (WIP) : only decoder training and with soft lip loss to all face mesh (not only lip vertices) but with epsilon 3.0 -> 3.5 (WIP)
python train_light.py --model_num "BEST_ict_vtx_v6" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## v7 (WIP): only decoder training and with soft lip loss to all face mesh (not only lip vertices) but with epsilon 3.5 -> 3.75 (WIP)
python train_light.py --model_num "BEST_ict_vtx_v7" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"
