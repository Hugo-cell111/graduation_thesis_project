python ./dataset_generate.py \
--train_val_data_dir "./datasets/CrackTree260/img/" \
--test_data_dir "./datasets/CRACK500/" \
--train_val_dataset_name "CrackTree260" \
--test_dataset_name "CRACK500" \
--train_dataset_ratio 0.99 \
--target1 "train_val" \
--threshold 0.5 \
--patch_size 256