# bert-base-cased
# bert-base-multilingual-cased
accelerate launch code/classification.py \
        --num_labels 2 \
        --model_name_or_path bert-base-multilingual-cased \
        --dataset CausalityDetection \
        --train_file data/Causality\ Detection/fnp2020-task1-train.csv \
        --test_file data/Causality\ Detection/fnp2020-task1-test.csv \
        --tokenizer_name bert-base-multilingual-cased \
        --max_seq_length 512 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 16 \
        --lr_bert 5e-6 \
        --lr_fc 1e-3 \
        --state 0 \
        --num_train_epochs 5 \
        --freeze_layer_count 0 \
        --seed 1000 \
        --with_tracking 