# train_FT
# step_three:FFA+caption  contrastive loss   text encoder + vision encoder
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run --nproc_per_node=4 --master_port='29502' ./pretraining/pretrain_FT_step2.py \
--modality "FFA" \
--epochs 20 --batch_size 24 --num_workers 4 \
--load_weights True --weights_path "./checkpoint/CFP_epoch20.pth" 

# train_RetCoP
# step_three:FFA+caption  contrastive loss + KL loss + buffer   text encoder + vision encoder
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run --nproc_per_node=4 --master_port='29502' ./pretraining/pretrain_RetCoP_step3.py \
--modality "FFA" \
--epochs 20 --batch_size 24 --num_workers 4 \
--load_weights True --weights_path "./checkpoint/CFP_epoch20.pth" 
