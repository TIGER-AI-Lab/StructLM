#!/bin/bash

__conda_setup="$('/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup

export HF_HOME=/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/.cache/huggingface
cd /ML-A100/team/mm/zhangge/gezhangmv/SKGLM/uskg_eval/
conda activate SKGLM
export WANDB_MODE=disabled

export NLTK_DATA=/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/nltk_data

python uskg_gen_dataset.py \
    --cfg new_data_gen.cfg \
    --run_name inst_T5_3b_all_tasks \
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
    --output_dir output/inst_T5_3b_all_tasks \
    --overwrite_output_dir \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --generation_num_beams 4 \
    --generation_max_length 512 \
    --report_to "tensorboard" \
    --input_max_length 2048 \
    --ddp_find_unused_parameters true \
    --gradient_checkpointing true
    # --do_eval \
    # --do_predict \
    # --metric_for_best_model avr \
    # --save_steps 10 \
    # --load_best_model_at_end \
    # --evaluation_strategy steps \
    # --eval_steps 10 \