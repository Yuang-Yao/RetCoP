# CFP   step two   

# zero-shot
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 0% --shots_test 100% --experiment 08_ODIR200x3 --method zero_shot --domain_knowledge True --project_features True --out_path "CFP_step2"

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 0% --shots_test 100% --experiment 13_FIVES --method zero_shot --domain_knowledge True --project_features True --out_path "CFP_step2"

# linear probe
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment 08_ODIR200x3 --method lp --domain_knowledge True --project_features False --out_path "CFP_step2"

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment 13_FIVES --method lp --domain_knowledge True --project_features False --out_path "CFP_step2"

# ClipAdapter
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment 08_ODIR200x3 --method clipAdapter --domain_knowledge True --project_features True --out_path "CFP_step2"

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment 13_FIVES --method clipAdapter --domain_knowledge True --project_features True --out_path "CFP_step2"





# CFP   step three   train_FT   

# zero-shot
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 0% --shots_test 100% --experiment 08_ODIR200x3 --method zero_shot --domain_knowledge True --project_features True --out_path "CFP_step3_train_FT" --weights_path ./checkpoint/train_FT_step3_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 0% --shots_test 100% --experiment 13_FIVES --method zero_shot --domain_knowledge True --project_features True --out_path "CFP_step3_train_FT" --weights_path ./checkpoint/train_FT_step3_epoch.pth

# linear probe
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment 08_ODIR200x3 --method lp --domain_knowledge True --project_features False --out_path "CFP_step3_train_FT" --weights_path ./checkpoint/train_FT_step3_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment 13_FIVES --method lp --domain_knowledge True --project_features False --out_path "CFP_step3_train_FT" --weights_path ./checkpoint/train_FT_step3_epoch.pth

# ClipAdapter
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment 08_ODIR200x3 --method clipAdapter --domain_knowledge True --project_features True --out_path "CFP_step3_train_FT" --weights_path ./checkpoint/train_FT_step3_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment 13_FIVES --method clipAdapter --domain_knowledge True --project_features True --out_path "CFP_step3_train_FT" --weights_path ./checkpoint/train_FT_step3_epoch.pth



# CFP   step three   train_RetCoP 

# zero-shot
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 0% --shots_test 100% --experiment 08_ODIR200x3 --method zero_shot --domain_knowledge True --project_features True --out_path "CFP_step3_train_RetCoP_epoch" --weights_path ./checkpoint/train_RetCoP_step3_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 0% --shots_test 100% --experiment 13_FIVES --method zero_shot --domain_knowledge True --project_features True --out_path "CFP_step3_train_RetCoP_epoch" --weights_path ./checkpoint/train_RetCoP_step3_epoch.pth

# linear probe
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment 08_ODIR200x3 --method lp --domain_knowledge True --project_features False --out_path "CFP_step3_train_RetCoP_epoch" --weights_path ./checkpoint/train_RetCoP_step3_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment 13_FIVES --method lp --domain_knowledge True --project_features False --out_path "CFP_step3_train_RetCoP_epoch" --weights_path ./checkpoint/train_RetCoP_step3_epoch.pth

# ClipAdapter
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment 08_ODIR200x3 --method clipAdapter --domain_knowledge True --project_features True --out_path "CFP_step3_train_RetCoP_epoch" --weights_path ./checkpoint/train_RetCoP_step3_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment 13_FIVES --method clipAdapter --domain_knowledge True --project_features True --out_path "CFP_step3_train_RetCoP_epoch" --weights_path ./checkpoint/train_RetCoP_step3_epoch.pth





# CFP   step four   train_FT   

# zero-shot
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 0% --shots_test 100% --experiment 08_ODIR200x3 --method zero_shot --domain_knowledge True --project_features True --out_path "CFP_step4_train_FT" --weights_path ./checkpoint/train_FT_step4_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 0% --shots_test 100% --experiment 13_FIVES --method zero_shot --domain_knowledge True --project_features True --out_path "CFP_step4_train_FT" --weights_path ./checkpoint/train_FT_step4_epoch.pth

# linear probe
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment 08_ODIR200x3 --method lp --domain_knowledge True --project_features False --out_path "CFP_step4_train_FT" --weights_path ./checkpoint/train_FT_step4_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment 13_FIVES --method lp --domain_knowledge True --project_features False --out_path "CFP_step4_train_FT" --weights_path ./checkpoint/train_FT_step4_epoch.pth

# ClipAdapter
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment 08_ODIR200x3 --method clipAdapter --domain_knowledge True --project_features True --out_path "CFP_step4_train_FT" --weights_path ./checkpoint/train_FT_step4_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment 13_FIVES --method clipAdapter --domain_knowledge True --project_features True --out_path "CFP_step4_train_FT" --weights_path ./checkpoint/train_FT_step4_epoch.pth



# CFP   step four   train_RetCoP

# zero-shot
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 0% --shots_test 100% --experiment 08_ODIR200x3 --method zero_shot --domain_knowledge True --project_features True --out_path "CFP_step4_train_RETCOP_epoch" --weights_path ./checkpoint/train_RETCOP_step4_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 0% --shots_test 100% --experiment 13_FIVES --method zero_shot --domain_knowledge True --project_features True --out_path "CFP_step4_train_RETCOP_epoch" --weights_path ./checkpoint/train_RETCOP_step4_epoch.pth

# linear probe
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment 08_ODIR200x3 --method lp --domain_knowledge True --project_features False --out_path "CFP_step4_train_RETCOP_epoch" --weights_path ./checkpoint/train_RETCOP_step4_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment 13_FIVES --method lp --domain_knowledge True --project_features False --out_path "CFP_step4_train_RETCOP_epoch" --weights_path ./checkpoint/train_RETCOP_step4_epoch.pth

# ClipAdapter
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment 08_ODIR200x3 --method clipAdapter --domain_knowledge True --project_features True --out_path "CFP_step4_train_RETCOP_epoch" --weights_path ./checkpoint/train_RETCOP_step4_epoch.pth

CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment 13_FIVES --method clipAdapter --domain_knowledge True --project_features True --out_path "CFP_step4_train_RETCOP_epoch" --weights_path ./checkpoint/train_RETCOP_step4_epoch.pth

