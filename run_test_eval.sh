# export NCCL_DEBUG=DEBUG
# export NCCL_IB_DISABLE=0
# export NCCL_IB_HCA=mlx5_1:1
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0

# __conda_setup="$('/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/miniconda3/etc/profile.d/conda.sh" ]; then
#         . "/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/miniconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/miniconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup

# export HF_HOME=/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/.cache/huggingface
# cd /ML-A100/team/mm/zhangge/gezhangmv/SKGLM/uskg_eval/
# conda activate SKGLM

# export WANDB_DISABLED=True


export NLTK_DATA=/ML-A100/team/mm/zhangge/gezhangmv/SKGLM/nltk_data

echo $NLTK_DATA

kwargs=" 
--overwrite_output_dir \
--per_device_eval_batch_size 64 \
--generation_max_length 1024 \
--learning_rate 2e-5 \
"

# CFG_PREFIX is the first argument
CFG_PREFIX=$1
INP_MAX_LEN=${2:-2048}


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
