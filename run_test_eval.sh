CFG_PREFIX=$1
INP_MAX_LEN=${2:-2176}
EVAL_BSIZE=${3:-64}

kwargs=" 
--overwrite_output_dir \
--per_device_eval_batch_size ${EVAL_BSIZE} \
--generation_max_length 1024 \
"

echo ""
echo "STARTING EVALUATION FOR: ${CFG_PREFIX}"
echo ""

# # if there is already a file named predictions_predict.json in the folder, say that
# if [ -f "output/${CFG_PREFIX}_${DATASET_NAME}/predictions_predict.json" ]; then
#     echo "output/${CFG_PREFIX}_${DATASET_NAME}/predictions_predict.json already exists, skipping"
#     exit 0
# fi

python eval.py $kwargs \
    --input_max_length ${INP_MAX_LEN} \
    --run_name ${CFG_PREFIX} \
    --output_dir output/${CFG_PREFIX} \
    --cfg ${CFG_PREFIX}.cfg \
    --vllm true

# run the evaluation script
python eval_json.py --run_name ${CFG_PREFIX}
