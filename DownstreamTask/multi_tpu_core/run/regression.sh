# bert-base-cased
# bert-base-multilingual-cased
accelerate launch DownstreamTask/multi_tpu_core/regression.py \
        --k_fold 10 \
        --model_name_or_path bert-base-multilingual-cased \
        --validation_strategy cross_validation \
        --data_file DownstreamTask/data/FiQA/task1/train/task1_headline_ABSA_train.json DownstreamTask/data/FiQA/task1/train/task1_post_ABSA_train.json \
        --tokenizer_name bert-base-multilingual-cased \
        --max_seq_length 512 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 16 \
        --optim_strategy 1 \
        --learning_rate 5e-5 \
        --lr_bert 5e-6 \
        --lr_fc 1e-3 \
        --num_train_epochs 10 \
        --freeze_layer_count 0 \
        --patience 3