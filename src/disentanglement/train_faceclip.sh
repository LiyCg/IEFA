## train faceclip 1st stage(this also covers ablation's all at once model)
## TO NOTE ## (deprecated)
## 1. This should involve lip loss but showing not about opening mouth case would still cover the experiment scope
## 2. This is based on original faceclip code, but using my own data, so no warmup neither
python train_faceclip.py --use_all_at_once True --model_num "ict_vtx_faceclip1_all_at_once_nolip" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

## SIG 2025

## /w LAFA retargeted data >> Failed (cause lip contacts are not maintained)
    ## TO NOTE ##
    ## 1. This had to be trained again for correct loss configuration of embedding loss of stage 2, since there are no trained stage 1 AE with lafa retargeted data ('--use_all_at_once False' for this)
    ## 2. So trained with default setting 
    ## trail 1 >> turned out the data was wrong
python train_faceclip.py --vtx_dtw_path "/input/inyup/IEFA/data/feature/lafa_dataset_m003_vtx_dtw_nolevel_v2.pickle" --model_num "lafa_ict_vtx_faceclip1_v2" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

    ## trail 2 
python train_faceclip.py --vtx_dtw_path "/input/inyup/IEFA/data/feature/lafa_dataset_m003_vtx_dtw_nolevel_v3.pickle" --model_num "lafa_ict_vtx_faceclip1_v3" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

    ## trial 3 w_tpl 0.001 -> 0.0005 (half) / --w_cross 10000 -> 8000
python train_faceclip.py --w_cross 8000 --w_tpl 0.0005 --model_num "lafa_ict_vtx_faceclip1_v4" --vtx_dtw_path "/input/inyup/IEFA/data/feature/lafa_dataset_m003_vtx_dtw_nolevel_v3.pickle" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

    ## trial 4 loss scale matched
python train_faceclip.py --w_self 10 --w_cross 10 --w_con 1 --w_tpl 3 --model_num "lafa_ict_vtx_faceclip1_v5" --vtx_dtw_path "/input/inyup/IEFA/data/feature/lafa_dataset_m003_vtx_dtw_nolevel_v3.pickle" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

    ## trial 5 loss scale matched
python train_faceclip.py --w_self 20 --w_cross 10 --w_con 1 --w_tpl 5 --model_num "lafa_ict_vtx_faceclip1_v6" --vtx_dtw_path "/input/inyup/IEFA/data/feature/lafa_dataset_m003_vtx_dtw_nolevel_v3.pickle" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

    ## trial 6 loss scale matched
python train_faceclip.py --w_self 10 --w_cross 10 --w_con 0.2 --w_tpl 0.2 --model_num "lafa_ict_vtx_faceclip1_v7" --vtx_dtw_path "/input/inyup/IEFA/data/feature/lafa_dataset_m003_vtx_dtw_nolevel_v3.pickle" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

    ## trial 7 loss scale matched
python train_faceclip.py --w_self 10 --w_cross 10 --w_con 0.2 --w_tpl 0.4 --model_num "lafa_ict_vtx_faceclip1_v8" --vtx_dtw_path "/input/inyup/IEFA/data/feature/lafa_dataset_m003_vtx_dtw_nolevel_v3.pickle" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

    ## trial 8 loss scale matched
python train_faceclip.py --w_self 20 --w_cross 10 --w_con 1 --w_tpl 4 --model_num "lafa_ict_vtx_faceclip1_v9"  --vtx_dtw_path "/input/inyup/IEFA/data/feature/lafa_dataset_m003_vtx_dtw_nolevel_v3.pickle" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"



## /w my own capture data but with the same preprocessing as Jung et al.
    ## TO NOTE ##
    ## 1. This is with gaussian filtered and dtw data 
    ## trail 1 
python train_faceclip.py --vtx_dtw_path "/input/inyup/IEFA/data/feature/ict_dataset_m003_vtx_dtw_gf_nolevel_03.pickle" --model_num "ict_vtx_faceclip1_v1" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

    ## trail 2
    ##  loss scale matching
python train_faceclip.py --model_num "ict_vtx_faceclip1_v2" --w_self 10 --w_cross 10 --w_con 1 --w_tpl 3 --vtx_dtw_path "/input/inyup/IEFA/data/feature/ict_dataset_m003_vtx_dtw_gf_nolevel_03.pickle" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"

    ## trail 3
    ##  loss scale matching with my data
python train_faceclip.py --model_num "ict_vtx_faceclip1_v3" --w_self 20 --w_cross 10 --w_con 1 --w_tpl 4 --vtx_dtw_path "/input/inyup/IEFA/data/feature/ict_dataset_m003_vtx_dtw_gf_nolevel_03.pickle" --vtx_dim 28227 --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy"
