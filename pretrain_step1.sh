# # step_two:CFP+caption   contrastive loss    text encoder + vision encoder
CUDA_VISIBLE_DEVICES=0,1,2,3 
python -m torch.distributed.run --nproc_per_node=4 --master_port='29502' ./pretraining/pretrain_CFP.py \
--epochs 40 --batch_size 24 --num_workers 4 \
--load_weights False --weights_path_report "./checkpoint/checkpoint-999.pth"