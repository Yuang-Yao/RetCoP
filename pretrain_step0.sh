# step_one:report   mlm loss   only text encoder
CUDA_VISIBLE_DEVICES=0,1,2,3 
python -m torch.distributed.run --nproc_per_node=4 --master_port='29502' ./pretraining/pretrain_report.py \
--data_root_path "/mnt/data/yayao/CPT_data/report/" \
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