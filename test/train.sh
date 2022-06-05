CUDA_VISIBLE_DEVICES="0" \
python -m torch.distributed.launch --master_port 3201 --nproc_per_node 1 train.py \
--model_name "model_one" \
--train_batch_size 64 \
--val_batch_size 1 \
--test_batch_size 1 \
--mode "test" \
--num_workers 4 \
--threshold 0.5 \
--eval_steps 50 \
--num_train_epochs 100 \
--lr 1e-4 \
--pos_weights 250 \
--loss_weight 0 \
--deep_supervision_weight 0.1 \
--pretrained_model "./model/model_one_epoch99_0418.pth"