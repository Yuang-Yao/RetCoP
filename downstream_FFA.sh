# FFA   step three   train_FT   

# zero-shot
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 0% --shots_test 100% --experiment MPOS --method zero_shot --domain_knowledge True --project_features True --out_path "FFA_step3_train_FT" --weights_path ./checkpoint/train_FT_step3_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 0% --shots_test 100% --experiment Angiographic --method zero_shot --domain_knowledge True --project_features True --out_path "FFA_step3_train_FT" --weights_path ./checkpoint/train_FT_step3_epoch.pth

# linear probe
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 20% --shots_test 20% --folds 5 --experiment MPOS --method lp --domain_knowledge True --project_features False --out_path "FFA_step3_train_FT" --weights_path ./checkpoint/train_FT_step3_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 20% --shots_test 20% --folds 5 --experiment Angiographic --method lp --domain_knowledge True --project_features False --out_path "FFA_step3_train_FT" --weights_path ./checkpoint/train_FT_step3_epoch.pth

# ClipAdapter
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 20% --shots_test 20% --folds 5 --experiment MPOS --method clipAdapter --domain_knowledge True --project_features True --out_path "FFA_step3_train_FT" --weights_path ./checkpoint/train_FT_step3_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 20% --shots_test 20% --folds 5 --experiment Angiographic --method clipAdapter --domain_knowledge True --project_features True --out_path "FFA_step3_train_FT" --weights_path ./checkpoint/train_FT_step3_epoch.pth



# FFA   step three   train_RetCoP

# zero-shot
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 0% --shots_test 100% --experiment MPOS --method zero_shot --domain_knowledge True --project_features True --out_path "FFA_step3_train_RetCoP" --weights_path ./checkpoint/train_RetCoP_step3_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 0% --shots_test 100% --experiment Angiographic --method zero_shot --domain_knowledge True --project_features True --out_path "FFA_step3_train_RetCoP" --weights_path ./checkpoint/train_RetCoP_step3_epoch.pth

# linear probe
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 20% --shots_test 20% --folds 5 --experiment MPOS --method lp --domain_knowledge True --project_features False --out_path "FFA_step3_train_RetCoP" --weights_path ./checkpoint/train_RetCoP_step3_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 20% --shots_test 20% --folds 5 --experiment Angiographic --method lp --domain_knowledge True --project_features False --out_path "FFA_step3_train_RetCoP" --weights_path ./checkpoint/train_RetCoP_step3_epoch.pth

# ClipAdapter
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 20% --shots_test 20% --folds 5 --experiment MPOS --method clipAdapter --domain_knowledge True --project_features True --out_path "FFA_step3_train_RetCoP" --weights_path ./checkpoint/train_RetCoP_step3_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 20% --shots_test 20% --folds 5 --experiment Angiographic --method clipAdapter --domain_knowledge True --project_features True --out_path "FFA_step3_train_RetCoP" --weights_path ./checkpoint/train_RetCoP_step3_epoch.pth





# FFA   step four   train_FT   

# zero-shot
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 0% --shots_test 100% --experiment MPOS --method zero_shot --domain_knowledge True --project_features True --out_path "FFA_step4_train_FT" --weights_path ./checkpoint/train_FT_step4_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 0% --shots_test 100% --experiment Angiographic --method zero_shot --domain_knowledge True --project_features True --out_path "FFA_step4_train_FT" --weights_path ./checkpoint/train_FT_step4_epoch.pth

# linear probe
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 20% --shots_test 20% --folds 5 --experiment MPOS --method lp --domain_knowledge True --project_features False --out_path "FFA_step4_train_FT" --weights_path ./checkpoint/train_FT_step4_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 20% --shots_test 20% --folds 5 --experiment Angiographic --method lp --domain_knowledge True --project_features False --out_path "FFA_step4_train_FT" --weights_path ./checkpoint/train_FT_step4_epoch.pth

# ClipAdapter
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 20% --shots_test 20% --folds 5 --experiment MPOS --method clipAdapter --domain_knowledge True --project_features True --out_path "FFA_step4_train_FT" --weights_path ./checkpoint/train_FT_step4_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 20% --shots_test 20% --folds 5 --experiment Angiographic --method clipAdapter --domain_knowledge True --project_features True --out_path "FFA_step4_train_FT" --weights_path ./checkpoint/train_FT_step4_epoch.pth



# FFA   step four   train_RetCoP

# zero-shot
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 0% --shots_test 100% --experiment MPOS --method zero_shot --domain_knowledge True --project_features True --out_path "FFA_step4_train_RetCoP" --weights_path ./checkpoint/train_RetCoP_step4_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 0% --shots_test 100% --experiment Angiographic --method zero_shot --domain_knowledge True --project_features True --out_path "FFA_step4_train_RetCoP" --weights_path ./checkpoint/train_RetCoP_step4_epoch.pth

# linear probe
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 20% --shots_test 20% --folds 5 --experiment MPOS --method lp --domain_knowledge True --project_features False --out_path "FFA_step4_train_RetCoP" --weights_path ./checkpoint/train_RetCoP_step4_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 20% --shots_test 20% --folds 5 --experiment Angiographic --method lp --domain_knowledge True --project_features False --out_path "FFA_step4_train_RetCoP" --weights_path ./checkpoint/train_RetCoP_step4_epoch.pth

# ClipAdapter
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 20% --shots_test 20% --folds 5 --experiment MPOS --method clipAdapter --domain_knowledge True --project_features True --out_path "FFA_step4_train_RetCoP" --weights_path ./checkpoint/train_RetCoP_step4_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 20% --shots_test 20% --folds 5 --experiment Angiographic --method clipAdapter --domain_knowledge True --project_features True --out_path "FFA_step4_train_RetCoP" --weights_path ./checkpoint/train_RetCoP_step4_epoch.pth




















