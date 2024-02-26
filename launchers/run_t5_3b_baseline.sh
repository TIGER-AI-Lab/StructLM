#!/bin/bash
__conda_setup="$('/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

which conda
export HF_HOME=/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/.cache/huggingface
cd /cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/
export WANDB_MODE=disabled
conda activate SKGLM

deepspeed train.py \
    --deepspeed deepspeed/ds_config_zero2.json \
    --seed 2 \
    --cfg Salesforce/old/new_T5_3b_finetune_all_tasks_2upsample.cfg \
    --run_name new_T5_3b_all_tasks \
    --logging_strategy steps \
    --logging_first_step true \
    --logging_steps 4 \
    --greater_is_better true \
    --save_strategy epoch \
    --save_total_limit 3 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --adafactor false \
    --learning_rate 5e-5 \
    --do_train \
    --predict_with_generate \
    --output_dir output/new_T5_3b_all_tasks \
    --overwrite_output_dir \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --generation_num_beams 4 \
    --generation_max_length 256 \
    --report_to "tensorboard" \
    --input_max_length 1024 \
    --ddp_find_unused_parameters true \
    --gradient_checkpointing true
    # --do_eval \
    # --do_predict \
    # --metric_for_best_model avr \
    # --save_steps 10 \
    # --load_best_model_at_end \
    # --evaluation_strategy steps \
    # --eval_steps 10 \