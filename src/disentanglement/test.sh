# python test.py --vtx_dtw_path "data/feature/ict_dataset_m003_vtx_dtw_nolevel.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" --model_num "ict_10000" 
# python test.py --vtx_dtw_path "data/feature/ict_dataset_m003_vtx_dtw_nolevel.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" --model_num "ict_10000" 
# python test.py --vtx_dtw_path "data/feature/ict_dataset_m003_vtx_dtw_nolevel_woneut.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" --model_num "ict_vtx_dtw_10000" 
# python test.py --vtx_dtw_path "data/feature/ict_dataset_m003_vtx_dtw_nolevel_woneut_02.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" --model_num "ict_vtx_dtw_10000_02" 
# python test.py --vtx_dtw_path "data/feature/ict_dataset_m003_vtx_dtw_nolevel_woneut_03.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" --model_num "ict_vtx_dtw_10000_03" 
# python test.py --vtx_dtw_path "data/feature/ict_dataset_m003_vtx_dtw_nolevel_woneut_04.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" --model_num "ict_vtx_dtw_10000_04" 
# python test.py --vtx_dtw_path "data/feature/ict_dataset_m003_vtx_dtw_nolevel_woneut_03.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" --model_num "ict_vtx_dtw_10000_03_wcs1" 
# python test.py --vtx_dtw_path "data/feature/ict_dataset_m003_vtx_dtw_nolevel_woneut_04_gf.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" --model_num "ict_vtx_dtw_10000_03_wcs1_gf" 
# python test.py --vtx_dtw_path "data/feature/ict_dataset_m003_vtx_dtw_nolevel_woneut_04_gf.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" --model_num "ict_vtx_dtw_10000_03_wcs1_gf_default"
# python test.py --vtx_dtw_path "data/feature/ict_dataset_m003_vtx_dtw_nolevel_woneut_03.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" --model_num "ict_vtx_dtw_10000_03" 


## finetune with bshp acitvated data test
# python test.py --vtx_dtw_path "data/feature/ict_dataset_m003_vtx_dtw_nolevel_woneut_03.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" --model_num "ict_vtx_dtw_5000_03_gf_default_mid" 
# python test.py --vtx_dtw_path "data/feature/ict_dataset_m003_vtx_dtw_nolevel_woneut_03.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" --model_num "ict_vtx_dtw_5000_03_gf_default_mid_vanilla" 

## test comparison between naive blendshape blending between our blending network
# python test.py --vtx_dtw_path "data/feature/ict_dataset_m003_vtx_dtw_nolevel_03.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" --model_num "ict_vtx_bshpAct_facsAct_warmup_facs200" 

## SIG 2025

    ## LAFA trained 
python test.py --model_num "lafa_ict_vtx_faceclip1_v5"  --vtx_dtw_path "/data/feature/lafa_dataset_m003_vtx_dtw_nolevel_v3.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 
python test.py --model_num "lafa_ict_vtx_faceclip1_v6"  --vtx_dtw_path "/data/feature/lafa_dataset_m003_vtx_dtw_nolevel_v3.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 
    
    ## my captured 

python test.py --model_num "ict_vtx_faceclip1_v2"  --vtx_dtw_path "/data/feature/_dataset_m003_vtx_dtw_nolevel_v3.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 
python test.py --model_num "ict_vtx_faceclip1_v3"  --vtx_dtw_path "/data/feature/lafa_dataset_m003_vtx_dtw_nolevel_v3.pickle" --neutral_vtx_file "ict_M003_front_neutral_1_011_last_fr.npy" 
    
