# OCT   step four   train_FT   

# zero-shot
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_OCT.py --shots_train 0% --shots_test 100% --experiment OCT05_OCTID --method zero_shot --domain_knowledge True --project_features True --out_path "OCT_step4_train_FT" --weights_path ./checkpoint/train_FT_step4_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_OCT.py --shots_train 0% --shots_test 100% --experiment OCT10_OCTDL --method zero_shot --domain_knowledge True --project_features True --out_path "OCT_step4_train_FT" --weights_path ./checkpoint/train_FT_step4_epoch.pth

# linear probe
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_OCT.py --shots_train 20% --shots_test 20% --folds 5 --experiment OCT05_OCTID --method lp --domain_knowledge True --project_features False --out_path "OCT_step4_train_FT" --weights_path ./checkpoint/train_FT_step4_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_OCT.py --shots_train 20% --shots_test 20% --folds 5 --experiment OCT10_OCTDL --method lp --domain_knowledge True --project_features False --out_path "OCT_step4_train_FT" --weights_path ./checkpoint/train_FT_step4_epoch.pth

# ClipAdapter
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_OCT.py --shots_train 20% --shots_test 20% --folds 5 --experiment OCT05_OCTID --method clipAdapter --domain_knowledge True --project_features True --out_path "OCT_step4_train_FT" --weights_path ./checkpoint/train_FT_step4_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_OCT.py --shots_train 20% --shots_test 20% --folds 5 --experiment OCT10_OCTDL --method clipAdapter --domain_knowledge True --project_features True --out_path "OCT_step4_train_FT" --weights_path ./checkpoint/train_FT_step4_epoch.pth



# OCT   step four   train_RetCoP

# zero-shot
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_OCT.py --shots_train 0% --shots_test 100% --experiment OCT05_OCTID --method zero_shot --domain_knowledge True --project_features True --out_path "OCT_step4_train_RetCoP" --weights_path ./checkpoint/train_RetCoP_step4_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_OCT.py --shots_train 0% --shots_test 100% --experiment OCT10_OCTDL --method zero_shot --domain_knowledge True --project_features True --out_path "OCT_step4_train_RetCoP" --weights_path ./checkpoint/train_RetCoP_step4_epoch.pth

# linear probe
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_OCT.py --shots_train 20% --shots_test 20% --folds 5 --experiment OCT05_OCTID --method lp --domain_knowledge True --project_features False --out_path "OCT_step4_train_RetCoP" --weights_path ./checkpoint/train_RetCoP_step4_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_OCT.py --shots_train 20% --shots_test 20% --folds 5 --experiment OCT10_OCTDL --method lp --domain_knowledge True --project_features False --out_path "OCT_step4_train_RetCoP" --weights_path ./checkpoint/train_RetCoP_step4_epoch.pth

# ClipAdapter
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_OCT.py --shots_train 20% --shots_test 20% --folds 5 --experiment OCT05_OCTID --method clipAdapter --domain_knowledge True --project_features True --out_path "OCT_step4_train_RetCoP" --weights_path ./checkpoint/train_RetCoP_step4_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_OCT.py --shots_train 20% --shots_test 20% --folds 5 --experiment OCT10_OCTDL --method clipAdapter --domain_knowledge True --project_features True --out_path "OCT_step4_train_RetCoP" --weights_path ./checkpoint/train_RetCoP_step4_epoch.pth



