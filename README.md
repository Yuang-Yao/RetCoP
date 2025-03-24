# Continual Retinal Vision-Language Pre-training upon Incremental Imaging Modalities


## 🌈 Quick Start
### 1. Environment
Clone the whole repository and install the dependencies.

- Python 3.8.18
- PyTorch 1.13.1
- cuda 12.0

```bash
conda create -n retcop python=3.8
conda activate retcop
pip install -r requirements.txt
```

### 2. Training

**Step 0:**  
Train the text encoder using medical report data.  
You can run the following command for training:  
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 
python -m torch.distributed.run --nproc_per_node=4 --master_port='29502' ./pretraining/pretrain_report.py \
--data_root_path "your_data_path" \
--batch_size 24 \
--epochs 1000 \
--warmup_epoch 40 \
--lr 1e-5 \ 
--weight_decay 0.05 \
--num_workers 10 \
--store_num 100 \
--mask_ratio 0.75 \
--out_path ./output/pretraining \
--log_dir ./output/pretraining 
```
Alternatively, you can directly run the following script:
```bash
sh ./pretrain_step0.sh
``` 
 
**Step 1:**  
Initialize the text encoder with the weights from Step 0, then train the text and visual encoders using CFP modality image-text pairs.
You can run the following command for training:
```bash
python -m torch.distributed.run --nproc_per_node=4 --master_port='29502' ./pretraining/pretrain_CFP.py \
--epochs 40 --batch_size 24 --num_workers 4 \
--load_weights False --weights_path_report "./checkpoint/checkpoint-999.pth"
```
Alternatively, you can directly run the following script:
```bash
sh ./pretrain_step1.sh
```
 
**Step 2:**  
Load the weights from Step 1, then continue training the text and visual encoders using FFA modality image-text pairs.
You can run the following command for training:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run --nproc_per_node=4 --master_port='29502' ./pretraining/pretrain_RetCoP_step3.py \
--modality "FFA" \
--epochs 20 --batch_size 24 --num_workers 4 \
--load_weights True --weights_path "./checkpoint/CFP_epoch20.pth" 
```
Alternatively, you can directly run the following script:
```bash
sh ./pretrain_step2.sh
``` 

**Step 3:**  
Load the weights from Step 2, then continue training the text and visual encoders using OCT modality image-text pairs.
You can run the following command for training:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run --nproc_per_node=4 --master_port='29502' ./pretraining/train_RetCoP_OCT.py \
--modality "OCT" \
--epochs 20 --batch_size 24 --num_workers 4 \
--load_weights True --weights_path "./checkpoint/train_RetCoP_step3_epoch.pth" 
```
Alternatively, you can directly run the following script:
```bash
sh ./pretrain_step3.sh
```

Note: 
Replace the data paths in the commands with your own paths. Also, ensure that the paths in the constants.py file are updated accordingly.

### 3. Evaluation

Downstream task transfer methods include ​zero-shot, ​linear probe, and ​clip-adapter.

**CFP**  
Evaluation can be performed at Step 1, Step 2, and Step 3. Run the following commands:

* zero-shot
```bash
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 0% --shots_test 100% --experiment your_dataset --method zero_shot --domain_knowledge True --project_features True --out_path "your_output_name" --weights_path your_weights_path
```

* linear probe
```bash
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment your_dataset --method lp --domain_knowledge True --project_features False --out_path "your_output_name" --weights_path your_weights_path
```

* clip-adapter
```bash
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_CFP.py --shots_train 20% --shots_test 20% --folds 5 --experiment  your_dataset --method clipAdapter --domain_knowledge True --project_features True --out_path "your_output_name" --weights_path your_weights_path
```

**FFA**  
Evaluation can be performed at Step 2, and Step 3. Run the following commands:

* zero-shot
```bash
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 0% --shots_test 100% --experiment your_dataset --method zero_shot --domain_knowledge True --project_features True --out_path "your_output_name" --weights_path your_weights_path
```

* linear probe
```bash
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 20% --shots_test 20% --folds 5 --experiment your_dataset --method lp --domain_knowledge True --project_features False --out_path "your_output_name" --weights_path your_weights_path
```

* clip-adapter
```bash
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_FFA.py --shots_train 20% --shots_test 20% --folds 5 --experiment your_dataset --method clipAdapter --domain_knowledge True --project_features True --out_path "your_output_name" --weights_path your_weights_path
```
 
**OCT**  
Evaluation can be performed at Step 3. Run the following commands:

* zero-shot
```bash
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_OCT.py --shots_train 0% --shots_test 100% --experiment your_dataset --method zero_shot --domain_knowledge True --project_features True --out_path "your_output_name" --weights_path your_weights_path
```

* linear probe
```bash
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_OCT.py --shots_train 20% --shots_test 20% --folds 5 --experiment your_dataset --method lp --domain_knowledge True --project_features False --out_path "your_output_name" --weights_path your_weights_path
```

* clip-adapter
```bash
CUDA_VISIBLE_DEVICES=0 python ./downstream/main_transferability_OCT.py --shots_train 20% --shots_test 20% --folds 5 --experiment your_dataset --method clipAdapter --domain_knowledge True --project_features True --out_path "your_output_name" --weights_path your_weights_path
```


## 💘 Acknowledge
FLAIR -- https://github.com/jusiro/FLAIR

FFA-IR -- https://github.com/mlii0117/FFA-IR

SynFundus-1M -- https://github.com/parap1uie-s/SynFundus-1M

MM-Retinal -- https://github.com/lxirich/MM-Retinal

MedCoSS -- https://github.com/yeerwen/medcoss
