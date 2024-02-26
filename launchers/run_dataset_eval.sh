export CUDA_VISIBLE_DEVICES=4
export WANDB_MODE=disabled
#export WANDB_MODE=offline

# conda init

# # conda activate alpaca

# __conda_setup="$('/home/alex/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/home/alex/miniconda3/etc/profile.d/conda.sh" ]; then
#         . "/home/alex/miniconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/home/alex/miniconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup

# conda activate alpaca

# echo $CUDA_VISIBLE_DEVICES

# nvidia-smi

#/home/alex/v3-score/llama_data_v2.json \
torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py \
    --run_name Llama_5ep_spider \
    --logging_strategy steps \
    --logging_first_step true \
    --logging_steps 4 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --metric_for_best_model avr \
    --greater_is_better true \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 1 \
    --load_best_model_at_end \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 400 \
    --adafactor true \
    --learning_rate 5e-5 \
    --do_predict \
    --predict_with_generate \
    --output_dir output/Llama_5ep_spider \
    --overwrite_output_dir \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --generation_num_beams 4 \
    --generation_max_length 128 \
    --input_max_length 1024 \
    --ddp_find_unused_parameters true \
    --cfg Salesforce/Llama_5ep_spider.cfg
    #--cfg Salesforce/Llama2_chat_5e_cosql.cfg
    #--cfg Salesforce/Llama2_finetune_5epochs_noinstr_cosql.cfg
    #--cfg Salesforce/Llama_finetune_5epochs_cosql.cfg
    #--cfg Salesforce/Llama_finetune_cosql.cfg \