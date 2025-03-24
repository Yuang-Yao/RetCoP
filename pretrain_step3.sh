# train_FT
# step_four:OCT+caption  contrastive loss   text encoder + vision encoder
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run --nproc_per_node=4 --master_port='29502' ./pretraining/train_FT_OCT.py \
--modality "OCT" \
--epochs 20 --batch_size 16 --num_workers 4 \
--load_weights True --weights_path "./checkpoint/train_FT_step3_epoch.pth" 

# train_RetCoP
# step_four:OCT+caption  contrastive loss + KL loss + buffer   text encoder + vision encoder
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.run --nproc_per_node=4 --master_port='29502' ./pretraining/train_RetCoP_OCT.py \
--modality "OCT" \
--epochs 20 --batch_size 24 --num_workers 4 \
--load_weights True --weights_path "./checkpoint/train_RetCoP_step3_epoch.pth" 

