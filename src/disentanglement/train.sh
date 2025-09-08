# python train.py --model_num "ict_default_5000" --vtx_dim 28224 --vtx_dtw_path "/source/inyup/TeTEC/faceClip/data/feature/ict_dataset_m003_vtx_dtw_nolevel.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 
# python train.py --epochs 10000 --model_num "ict_10000" --vtx_dim 28227 --vtx_dtw_path "/source/inyup/TeTEC/faceClip/data/feature/ict_dataset_m003_vtx_dtw_nolevel.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 
# python train.py --epochs 10000 --model_num "ict_vtx_dtw_10000" --vtx_dim 28227 --vtx_dtw_path "/source/inyup/TeTEC/faceClip/data/feature/ict_dataset_m003_vtx_dtw_nolevel_woneut.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 
# python train.py --epochs 10000 --model_num "ict_vtx_dtw_10000_02" --vtx_dim 28227 --vtx_dtw_path "/source/inyup/TeTEC/faceClip/data/feature/ict_dataset_m003_vtx_dtw_nolevel_woneut_02.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 
# python train.py --epochs 10000 --model_num "ict_vtx_dtw_10000_03" --vtx_dim 28227 --vtx_dtw_path "/source/inyup/TeTEC/faceClip/data/feature/ict_dataset_m003_vtx_dtw_nolevel_woneut_03.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 
# python train.py --epochs 10000 --model_num "ict_vtx_dtw_10000_04" --vtx_dim 28227 --vtx_dtw_path "/source/inyup/TeTEC/faceClip/data/feature/ict_dataset_m003_vtx_dtw_nolevel_woneut_04.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 
# python train.py --lr 1e-6 --epochs 10000 --model_num "ict_vtx_dtw_10000_03_lr06" --vtx_dim 28227 --vtx_dtw_path "/source/inyup/TeTEC/faceClip/data/feature/ict_dataset_m003_vtx_dtw_nolevel_woneut_03.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 
# python train.py --w_cross 1 --w_self 1 --epochs 10000 --model_num "ict_vtx_dtw_10000_03_wcs1" --vtx_dim 28227 --vtx_dtw_path "/source/inyup/TeTEC/faceClip/data/feature/ict_dataset_m003_vtx_dtw_nolevel_woneut_03.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 

## with gaussian filtered data,
# python train.py --w_cross 1 --w_self 1 --epochs 10000 --model_num "ict_vtx_dtw_10000_03_wcs1_gf" --vtx_dim 28227 --vtx_dtw_path "/input/inyup/IEFA/data/feature/ict_dataset_m003_vtx_dtw_nolevel_woneut_04_gf.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 
# python train.py --epochs 5000 --model_num "ict_vtx_dtw_10000_03_wcs1_gf_default" --vtx_dim 28227 --vtx_dtw_path "/input/inyup/IEFA/data/feature/ict_dataset_m003_vtx_dtw_nolevel_woneut_04_gf.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 


## fine-tune with individual blendshape activated data
# python train.py --epochs 2500 --model_num "ict_vtx_dtw_10000_03_wcs1_gf_default_mid" --vtx_dim 28227 --vtx_dtw_path "/input/inyup/IEFA/data/feature/ict_dataset_m003_vtx_bshpAct_01.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 


## curriculum training
# 2000 / 3000 >> recognize single eye region expression, but not faithfully decodes
# python train.py --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 
# 1500 / 2000 / 3000 >> doesn't recognize single eye region expression at all
## warm up applied
# python train.py --use_warmup True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 
# 3000 / 2000 / 3000 >> does recognize single eye, but both eyes react and they don't close completley
# python train.py --use_warmup True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 
# 3000 / 2000 / 1000 >> does recognize single eye, but still the other eye responds, though with a little less intensity + mouth wont open as expected
# python train.py --use_warmup True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup_facs1000" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 
# 3000 / 2000 >> perfectly recognize and decodes single eye or any single expression but not able to make multiple targets into a single expression
# python train.py --use_warmup True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup_facs500" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 

## facs200 version finetuned with warm up 2 
# 3000 / 2000 / 200 / 500 >> 
# python train.py --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup_facs200_warmup500" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"  # compromised the previous latent(blinking degraded), mouth dimple, mouth_lower_l still doesn't work  
# 3000 / 2000 / 200 / 1000 >>
# python train.py --warmup2_epochs 1000 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup_facs200_warmup1000" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" # compromised more 
# 3000 / 2000 / 200 / 1500 >> 
# python train.py --warmup2_epochs 1500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup_facs200_warmup1500" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" # deleted
# 3000 / 2000 / 200 / 2000 >>
# python train.py --warmup2_epochs 2000 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup_facs200_warmup2000" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" # deleted

## warm up at very front 
# 250 / 3000 / 2000 / 200 
# python train.py --warmup2_epochs 250 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup250_warmup_facs200" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"  # compromised the previous latent(blinking degraded), mouth dimple, mouth_lower_l still doesn't work  
# 500 / 3000 / 2000 / 200
# python train.py --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"  # compromised the previous latent(blinking degraded), mouth dimple, mouth_lower_l still doesn't work  
# 750 / 3000 / 2000 / 200
# python train.py --warmup2_epochs 750 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup750_warmup_facs200" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"  # compromised the previous latent(blinking degraded), mouth dimple, mouth_lower_l still doesn't work  
# 1000 / 3000 / 2000 / 200
# python train.py --warmup2_epochs 1000 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup1000_warmup_facs200" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"  # compromised the previous latent(blinking degraded), mouth dimple, mouth_lower_l still doesn't work  


## with lip contact loss
# python train.py --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_liploss" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"  
## with lip contact loss but without mse with target
# python train.py --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_liploss_nomse" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"  
## with lip contact loss but with mse 0.01 # accidently trained more with mse 0.1
# python train.py --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_liploss_mse001" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"  
## with lip contact loss but with lambda value 0.01
# python train.py --w_lip 0.01 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_liploss_ld001" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"  
## with lip contact loss but with lambda value 0.1
# python train.py --w_lip 0.1 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_liploss_ld01" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"  
## with lip contact loss but with mse 0.1
# python train.py --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_liploss_mse01" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"  
## with lip contact loss but with lamda value 0.05 (between 0.1 and 0.01)
# python train.py --w_lip 0.05 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_liploss_ld005" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 
## with lip contact loss but with lamda value 0.025 (between 0.1 and 0.05)
# python train.py --w_lip 0.025 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_liploss_ld0025" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 
## with lip contact loss but with lamda value 0.075 (between 0.1 and 0.05)
# python train.py --w_lip 0.075 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_liploss_ld0075" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 

## with new lip contact loss (considering lip dynamics)
## with lip contact loss but with ld 0.1 loose_scale to 0.1
# python train.py --w_lip 0.1 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_newliploss_ld01" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"  
## with lip contact loss but with ld 0.1 but loose_scale to 0.4
# python train.py --w_lip 0.1 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_newliploss_ld01_loo04" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"  
## with lip contact loss but with ld 0.1 but loose_scale to 0.01
# python train.py --w_lip 0.1 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_newliploss_ld01_loo001" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"  
## with lip contact loss but with ld 0.1 but loose_scale to 0.25
# python train.py --w_lip 0.1 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_newliploss_ld01_loo025" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"  
## with lip contact loss but with ld 0.1 but loose_scale to 0.175
# python train.py --w_lip 0.1 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_newliploss_ld01_loo0175" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"  
## with lip contact loss but with ld 0.1 but loose_scale to 0.1375
# python train.py --w_lip 0.1 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_newliploss_ld01_loo01375" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"  
## with lip contact loss but with ld 0.1 but loose_scale to 0.15625
# python train.py --w_lip 0.1 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_newliploss_ld01_loo015625" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"
## with lip contact loss but with ld 0.1 but loose_scale to 0.165625
# python train.py --w_lip 0.1 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_newliploss_ld01_loo0165625" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"    
## with lip contact loss but with ld 0.1 but loose_scale to 0.17
# python train.py --w_lip 0.1 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_newliploss_ld01_loo017" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"    
## with lip contact loss but with ld 0.1 but loose_scale to 0.1675 (BEST)
# python train.py --w_lip 0.1 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_newliploss_ld01_loo01675" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"    

## with lip contact loss but with ld 0.1 but loose_scale to 0.1675 with epsilon 0.0001 / (orig: 0.01)
# python train.py --w_lip 0.1 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_newliploss_ld01_loo01675_e0001" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"    
## with lip contact loss but with ld 0.1 but loose_scale to 0.1675 with epsilon 0.001 / (orig: 0.01)
# python train.py --w_lip 0.1 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_newliploss_ld01_loo01675_e001" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"    
## with lip contact loss but with ld 0.1 but loose_scale to 0.1675 with epsilon 0.005 / (orig: 0.01)
# python train.py --w_lip 0.1 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_newliploss_ld01_loo01675_e005" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## with lip contact loss but but loose_scale to 0.1675(with best) with epsilon 0.01(back to orig) with ld lip 1.0 
# python train.py --w_lip 1.0 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshpAct_facsAct_warmup500_warmup_facs200_newliploss_ld1_loo01675" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## with emotion 50 lip contact loss, loose_scale to 0.1675(with best) with epsilon 0.01(back to orig) with ld lip 10 (best)
# python train.py --w_lip 10 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_bshp_facs_emot_warmup500_warmup_facs200_emot100_newlip_ld10_loo01675" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## with emotion act lip contact loss, loose_scale to 0.1675(with best) with epsilon 0.01(back to orig) with ld lip 10 but without warmup2
# python train.py --w_lip 10 --use_lip_contact_loss True --use_warmup True --use_curriculum_training True --model_num "ict_vtx_bshp_facs_emot_warmup_facs200_emot50_newlip_ld10_loo01675" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## with emotion act lip contact loss but but loose_scale to 0.1675(with best) with epsilon 0.01(back to orig) with ld lip 10 but without warmup2 with emot 100
# python train.py --w_lip 10 --use_lip_contact_loss True --use_warmup True --use_curriculum_training True --model_num "ict_vtx_bshp_facs_emot_warmup_facs200_emot50_newlip_ld10_loo01675" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## (GOAT - b4 lip loss) with emotion 50 lip contact loss loose_scale to 0.1675(with best) with epsilon 3.8 with ld lip 10 but with L1 weaker loss 
# python train.py --w_lip 10 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_warmup2_warmup_bshp_facs200_emot50_l1lip_ld10_loo01675" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## with emotion 50 lip contact loss loose_scale to 0.1675(with best) with epsilon 3.8 with ld lip 10
## same with just above but set 'use_mse' in 'model_AE.py' def lip_contact_loss() func to True
# python train.py --w_lip 10 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_warmup2_warmup_bshp_facs200_emot50_l2lip_ld10_loo01675" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## with emotion 50 lip contact loss loose_scale to 0.1675(with best) with epsilon 3.8 with ld lip 10 but both stong and loose lip loss l1 loss
# python train.py --w_lip 10 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_warmup2_warmup_bshp_facs200_emot50_2l2lip_ld10_loo01675" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## version 3 lip loss with epsilon 3.8
## python train.py --w_lip 10 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_warmup2_warmup_bshp_facs200_emot50_2l2lip3_ld10_loo01675" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## version 3 lip loss with epsilon 4.2
## python train.py --w_lip 10 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_warmup2_warmup_bshp_facs200_emot50_2l2lip3_ld10_loo01675_ep42" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## version 3 lip loss with epsilon 4.0
python train.py --w_lip 10 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_warmup2_warmup_bshp_facs200_emot50_2l2lip3_ld10_loo01675_ep40" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## version 4 lip loss with epsilon 4.0 loose 1.0 both l1
python train.py --w_lip 10 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_warmup2_warmup_bshp_facs200_emot50_2l1lip4_ld10_loo10_ep40" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"
## version 4 lip loss with epsilon 4.0 loose 1.0 only contact case loss l1
python train.py --w_lip 10 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_warmup2_warmup_bshp_facs200_emot50_1l1lip4_ld10_loo10_ep40" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"
## version 4 lip loss with epsilon 4.0 loose 0.1675 only contact case loss l1 (BEST for overall result) -> but have artifacts of bulging and protruding of selected lip landmarks 
## (BEST)python train.py --w_lip 10 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_warmup2_warmup_bshp_facs200_emot50_1l1lip4_ld10_loo1675_ep40" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"
## version 4 lip loss with epsilon 3.8 loose 0.1675 only contact case loss l1 (was best) -> but not for this version 4 case
python train.py --w_lip 10 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_warmup2_warmup_bshp_facs200_emot50_1l1lip4_ld10_loo1675_ep38" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## soft mask lip loss version with version 4's BEST setup (with epsilon 4.0 loose 0.1675 only contact case loss l1) -> still bulging but lessened but, not following expression 
python train.py --use_lip_contact_loss True --use_soft_lip_contact_loss True --use_warmup2 True --use_curriculum_training True --w_lip 10 --warmup2_epochs 500  --model_num "ict_vtx_warmup2_warmup_bshp_facs200_emot50_1l1lip4_ld10_loo1675_ep40_soft" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"
## soft mask lip loss version with version 4's BEST setup (with epsilon 4.0 loose 0.1675 'smoothing factor 0.1 -> 0.6' only contact case loss l1) -> 
python train.py --use_lip_contact_loss True --use_soft_lip_contact_loss True --use_warmup2 True --use_curriculum_training True --w_lip 10 --warmup2_epochs 500  --model_num "ict_vtx_warmup2_warmup_bshp_facs200_emot50_1l1lip4_ld10_loo1675_ep40_soft_sm06" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"
## soft mask lip loss version with version 4's BEST setup (with epsilon 4.0 loose 0.1675 'smoothing factor 0.1 -> 0.3' only contact case loss l1) -> 
python train.py --use_lip_contact_loss True --use_soft_lip_contact_loss True --use_warmup2 True --use_curriculum_training True --w_lip 10 --warmup2_epochs 500  --model_num "ict_vtx_warmup2_warmup_bshp_facs200_emot50_1l1lip4_ld10_loo1675_ep40_soft_sm03" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"
## laplacian added + soft mask lip loss version with version 4's BEST setup (with epsilon 4.0 loose 0.1675 'smoothing factor 0.1 -> 0.3' only contact case loss l1) -> 
python train.py --use_lap True --use_lip_contact_loss True --use_soft_lip_contact_loss True --use_warmup2 True --use_curriculum_training True --w_lip 10 --warmup2_epochs 500  --model_num "ict_vtx_warmup2_warmup_bshp_facs200_emot50_1l1lip4_ld10_loo1675_ep40_soft_sm03_lap" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## laplacian added ld 0.1 + soft mask lip loss version with version 4's BEST setup (with epsilon 4.0 loose 0.1675 'smoothing factor 0.1 -> 0.3' only contact case loss l1) -> 
python train.py --use_lap True --use_lip_contact_loss True --use_soft_lip_contact_loss True --use_warmup2 True --use_curriculum_training True --w_lip 10 --warmup2_epochs 500  --model_num "ict_vtx_warmup2_warmup_bshp_facs200_emot50_1l1lip4_ld10_loo1675_ep40_soft_sm03_lap_01" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"


## no lip loss 
python train.py --use_lip_contact_loss False --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_warmup2_warmup_bshp_facs200_emot50_nolip" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

### ABLATION
## reverse order 
python train.py --use_reverse_order True --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "ict_vtx_warmup2_warmup_bshp_facs200_emot50_reverse" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"


####################
## SIG2025 #########

## v1 : with content encoder also takes gradient of lip contact loss
python train.py --use_train_con True --w_lip 10 --use_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "BEST_ict_vtx_v1" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## v2 : with no lip loss for the whole epoch
python train.py --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "BEST_ict_vtx_v2" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## v3 : only decoder training and with soft lip loss to all face mesh (not only lip vertices)
python train.py --w_lip 10 --use_lip_contact_loss True --use_soft_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "BEST_ict_vtx_v3" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## v4 : only decoder training and with soft lip loss to all face mesh (not only lip vertices) but with epsilon 4.0 -> 2.0
python train.py --w_lip 10 --use_lip_contact_loss True --use_soft_lip_contact_loss True --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --model_num "BEST_ict_vtx_v4" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## v5 : only decoder training and with soft lip loss to all face mesh (not only lip vertices) but with epsilon 2.0 -> 3.0 >> still gaping
python train.py --model_num "BEST_ict_vtx_v5" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## v6 (with light WIP) : only decoder training and with soft lip loss to all face mesh (not only lip vertices) but with epsilon 3.0 -> 3.5 (WIP)
python train.py --model_num "BEST_ict_vtx_v6" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## v7 (with light WIP) : only decoder training and with soft lip loss to all face mesh (not only lip vertices) but with epsilon 3.5 -> 3.75 (WIP)
python train.py --model_num "BEST_ict_vtx_v7" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --warmup2_epochs 500 --use_warmup2 True --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"


###############
## SIGA 2025 ##

## with different ROM considered for each bshp targets (everything else is same as v7)
# loose_scale=0.1675, epsilon=3.5, smoothing_factor=0.6, use_mse_strong=False, use_mse_loose=True
python train.py --model_num "ict_vtx_diffrom_v1" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## without Emotion and reverse(?? not sure) order (FACS - Blendshape) - accidentally did this;; 
# group.add_argument("--bshpAct_data_dir", type=str, default="/input/inyup/IEFA/data/feature/ict_dataset_m003_vtx_bshpAct_03_neut.pickle")
# group.add_argument("--facsAct_data_dir", type=str, default="/input/inyup/IEFA/data/feature/ict_dataset_m003_vtx_facsAct_03_neut.pickle")
# group.add_argument("--emotion_data_dir", type=str, default="/input/inyup/IEFA/data/feature/ict_dataset_m003_vtx_emotAct_03_neut.pickle")
python train.py --model_num "ict_vtx_diffrom_v1_neut" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## with Emotion and correct order 
python train.py --model_num "ict_vtx_diffrom_v1_neut_ours" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## without Emotion and wrong order (same with ict_vtx_diffrom_v1)
python train.py --model_num "ict_vtx_diffrom_v1_neut_same" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## without Emotion and wrong order (exactly same with ict_vtx_diffrom_v1, with real ROM (이전엔 0.75씩 곱해져있었음음), with a little bit of neutral to bshp training)
python train.py --model_num "ict_vtx_diffrom_v1_neut_same-real" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## without Emotion and wrong order (exactly same with ict_vtx_diffrom_v1, with real ROM but with Econ training as well(이전엔 0.75씩 곱해져있었음음), with a little bit of neutral to bshp training)
python train.py --model_num "ict_vtx_diffrom_v1_neut_same-real_1" --use_train_con True --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## with different ROM considered for each bshp targets (everything else is same as v7) also trains contenct encoder (Econ)
# loose_scale=0.1675, epsilon=3.5, smoothing_factor=0.6, use_mse_strong=False, use_mse_loose=True 
python train.py --model_num "ict_vtx_diffrom_v2" --use_train_con True --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## with different ROM considered for each bshp targets (everything else is same as v7) same with v1, but with more epochs for each HDFS category
# loose_scale=0.1675, epsilon=3.5, smoothing_factor=0.6, use_mse_strong=False, use_mse_loose=True (1.5 배씩)
python train.py --bshp_epochs 3000 --facs_epochs 300 --emot_epochs 75 --model_num "ict_vtx_diffrom_v3" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## without Emotion and correoct order 
## 이게 원래 ict_vtx_diffrom_v1 였음 + emot epoch만큼 더 FACS를 학습했던거였음 위에 real/real_1은) 
## exactly same with ict_vtx_diffrom_v1, with real ROM (이전엔 0.75씩 곱해져있었음음)
## with neutral to bshp training only, without eye_turn_* in facs
python train.py --model_num "ict_vtx_diffrom_v4" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"
## -> 데이터 상에 문제가 있는지, 오른쪽 눈이 이상하게 neutral로 잡힘. 

## without Emotion and correoct order 
## 이게 원래 ict_vtx_diffrom_v1 였음 + emot epoch만큼 더 FACS를 학습했던거였음 위에 real/real_1은) 
## exactly same with ict_vtx_diffrom_v1, with real ROM
## without neutral to bshp training only, without eye_turn_* in facs
python train.py --model_num "ict_vtx_diffrom_v5" --bshpAct_data_dir "/input/inyup/IEFA/data/feature/s1_ict_dataset_m003_vtx_bshpAct_03.pickle" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## without Emotion and correct order 
## 이게 원래 ict_vtx_diffrom_v1 였음 + emot epoch만큼 더 FACS를 학습했던거였음 위에 real/real_1은) 
## exactly same with ict_vtx_diffrom_v1, with real ROM
## -----------------------------------------------------------
## without neutral to bshp training only, without eye_turn_* in facs
## -> 원래 s1_..._facsAct 데이터셋에서, eye관련애들 빼고 다시 학습하기. (원래 v1에서 썼던 데이터셋 그대로에서 eye관련애들 삭제!)
python train.py --model_num "ict_vtx_diffrom_v6" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## without Emotion and correct order 
## 이게 원래 ict_vtx_diffrom_v1 였음 + emot epoch만큼 더 FACS를 학습했던거였음 위에 real/real_1은) 
## exactly same with ict_vtx_diffrom_v1, with real ROM
## -----------------------------------------------------------
## with neutral to bshp training only, without eye_turn_* in facs
## -> 원래 s1_..._facsAct 데이터셋에서, eye관련애들 빼고 다시 학습하기. (원래 v1에서 썼던 데이터셋 그대로에서 eye관련애들 삭제!) -> 안뺀애로 해버림..
## -> v6랑 command 자체는 똑같은데, parser_util.py에서 바꿔줬음
python train.py --model_num "ict_vtx_diffrom_v7" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## without Emotion and correct order 
## 이게 원래 ict_vtx_diffrom_v1 였음 
## -----------------------------------------------------------
## without neutral to bshp training only, with eye_turn_* in facs 
## -> v6, v7랑 command 자체는 똑같은데, parser_util.py에서 바꿔줬음
python train.py --model_num "ict_vtx_diffrom_v8" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"


## -----------------------------------------------------------
## v6,7,8 결과 보고 발전시킨 내용
## -----------------------------------------------------------


## without Emotion and correct order 
## 이게 원래 ict_vtx_diffrom_v1 였음 + emot epoch만큼 더 FACS를 학습했던거였음 위에 real/real_1은) 
## exactly same with ict_vtx_diffrom_v1, with real ROM
## -----------------------------------------------------------
## with neutral to bshp training only, without eye_turn_* in facs
## -> 원래 s1_..._facsAct 데이터셋에서, ""제대로""" eye관련애들 빼고 다시 학습하기. (원래 v1에서 썼던 데이터셋 그대로에서 eye관련애들 삭제!)
## -> v6, v7, v8랑 command 자체는 똑같은데, parser_util.py에서 바꿔줬음
# group.add_argument("--bshpAct_data_dir", type=str, default="/input/inyup/IEFA/data/feature/ict_dataset_m003_vtx_bshpAct_03_neut.pickle")
# group.add_argument("--facsAct_data_dir", type=str, default="/input/inyup/IEFA/data/feature/ict_dataset_m003_vtx_facsAct_03_woEyes.pickle") -> 또 그 눈깔 한쪽 잘못된 에러 발생 
python train.py --model_num "ict_vtx_diffrom_v9" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## without Emotion and correct order 
## 이게 원래 ict_vtx_diffrom_v1 였음 
## -----------------------------------------------------------
## without neutral to bshp training only, without eye_turn_* in facs 
## -> v6, v7랑 command 자체는 똑같은데, parser_util.py에서 바꿔줬음
# group.add_argument("--bshpAct_data_dir", type=str, default="/input/inyup/IEFA/data/feature/s1_ict_dataset_m003_vtx_bshpAct_03.pickle")
# group.add_argument("--facsAct_data_dir", type=str, default="/input/inyup/IEFA/data/feature/ict_dataset_m003_vtx_facsAct_03_woEyes.pickle") -> 또 그 눈깔 한쪽 잘못된 에러 발생 
python train.py --model_num "ict_vtx_diffrom_v10" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"


## -----------------------------------------------------------
## v9,10 결과 보고 발전시킨 내용 (둘다 왼쪽 눈 artifact 다시생김김)
## -----------------------------------------------------------


## without Emotion and correct order 
## 이게 원래 ict_vtx_diffrom_v1 였음 + emot epoch만큼 더 FACS를 학습했던거였음 위에 real/real_1은) 
## exactly same with ict_vtx_diffrom_v1, with real ROM
## -----------------------------------------------------------
## with neutral to bshp training only, without eye_turn_* in facs
## -> 원래 s1_..._facsAct 데이터셋에서, ""제대로""" eye관련애들 빼고 다시 학습하기. (원래 v1에서 썼던 데이터셋 그대로에서 eye관련애들 삭제!)
## -> v6, v7, v8랑 command 자체는 똑같은데, parser_util.py에서 바꿔줬음
# group.add_argument("--bshpAct_data_dir", type=str, default="/input/inyup/IEFA/data/feature/ict_dataset_m003_vtx_bshpAct_03_neut.pickle")
# group.add_argument("--facsAct_data_dir", type=str, default="/input/inyup/IEFA/data/feature/ict_dataset_m003_vtx_facsAct_03_woEyes-artifact.pickle") 
python train.py --model_num "ict_vtx_diffrom_v11" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## without Emotion and correct order 
## 이게 원래 ict_vtx_diffrom_v1 였음 
## -----------------------------------------------------------
## without neutral to bshp training only, without eye_turn_* in facs 
## -> v6, v7랑 command 자체는 똑같은데, parser_util.py에서 바꿔줬음
# group.add_argument("--bshpAct_data_dir", type=str, default="/input/inyup/IEFA/data/feature/s1_ict_dataset_m003_vtx_bshpAct_03.pickle")
# group.add_argument("--facsAct_data_dir", type=str, default="/input/inyup/IEFA/data/feature/ict_dataset_m003_vtx_facsAct_03_woEyes-artifact.pickle")
python train.py --model_num "ict_vtx_diffrom_v12" --use_lip_contact_loss True --use_soft_lip_contact_loss True --w_lip 10 --use_curriculum_training True --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

