export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=disabled

kwargs=" 
--overwrite_output_dir \
--per_device_eval_batch_size 1 \
--generation_max_length 256 \
--input_max_length 2048 \
--ddp_find_unused_parameters true \
"

# CFG_PREFIX is the first argument
CFG_PREFIX=$1
# example "non_upsampled_e2"
DATASET_NAME=$2
MASTER_PORT=$3
LLAMA_PORT=$4
# example "spider"


echo ""
echo "STARTING EVALUATION FOR: ${CFG_PREFIX}_${DATASET_NAME}"
echo ""

# # if there is already a file named predictions_predict.json in the folder, say that
# if [ -f "output/${CFG_PREFIX}_${DATASET_NAME}/predictions_predict.json" ]; then
#     echo "output/${CFG_PREFIX}_${DATASET_NAME}/predictions_predict.json already exists, skipping"
#     exit 0
# fi

torchrun --nproc_per_node 1 --master_port=$MASTER_PORT eval_llama_stripped.py $kwargs --run_name ${CFG_PREFIX}_${DATASET_NAME} --test_split_json processed_data/llama_data_v11_kg_upated_test.json --output_dir output/${CFG_PREFIX}_${DATASET_NAME} --cfg Salesforce/${CFG_PREFIX}_${DATASET_NAME}.cfg --port $LLAMA_PORT