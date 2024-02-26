export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled

kwargs=" --logging_strategy steps \
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
--overwrite_output_dir \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 16 \
--generation_num_beams 1 \
--generation_max_length 128 \
--input_max_length 1024 \
--ddp_find_unused_parameters true \
"   

# CFG_PREFIX is the first argument
CFG_PREFIX=$1
# example "non_upsampled_e2"
DATASET_NAME=$2
MASTER_PORT=$3
# example "spider"


echo ""
echo "STARTING EVALUATION FOR: ${CFG_PREFIX}_${DATASET_NAME}"
echo ""

# if there is already a file named predictions_predict.json in the folder, say that
if [ -f "output/${CFG_PREFIX}_${DATASET_NAME}/predictions_predict.json" ]; then
    echo "output/${CFG_PREFIX}_${DATASET_NAME}/predictions_predict.json already exists, skipping"
    exit 0
fi

torchrun --nproc_per_node 1 --master_port=$MASTER_PORT eval_t5.py $kwargs --run_name ${CFG_PREFIX}_${DATASET_NAME} --output_dir output/${CFG_PREFIX}_${DATASET_NAME} --cfg Salesforce/${CFG_PREFIX}_${DATASET_NAME}.cfg