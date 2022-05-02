# bert-base-cased
# bert-base-multilingual-cased
accelerate launch DownstreamTask/multi_tpu_core/classification.py \
        --num_labels 3 \
        --k_fold 5 \
        --model_name_or_path bert-base-multilingual-cased \
        --validation_strategy cross_validation \
        --dataset Lithuanian \
        --data_file DownstreamTask/data/LithuanianNewsSentiment/dataset.csv \
        --tokenizer_name bert-base-multilingual-cased \
        --max_seq_length 512 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 16 \
        --lr_bert 5e-6 \
        --lr_fc 1e-3 \
        --state 0 \
        --num_train_epochs 20 \
        --freeze_layer_count 0