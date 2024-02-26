# CUDA_VISIBLE_DEVICES=6 python llama_service.py \
#     --llama_path '/mnt/tjena/alex/llama2_base_no_instr_7_26' \
#     --port 8090\

    #--llama_path '/mnt/tjena/alex/llama_6_29' \
# -m pdb -c "c"
    
#--model_path '/mnt/tjena/alex/vaughan/checkpoint-15380' \
#--model_path '/mnt/tjena/alex/llama2_7_26' \
#--model_path '/mnt/tjena/alex/vaughan/xgen/checkpoint-10253' \
#--model_path '/mnt/tjena/alex/vaughan/codellamae5' \

# command line args for the model path and the port

# /cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/models/llama/v7_non_upsampled/checkpoint-2362
# /cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/models/llama/v7_non_upsampled/checkpoint-4724
# /cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/models/llama/v7_non_upsampled/checkpoint-7086

MODEL_PATH=$2
PORT=$3

python llama_service.py \
    --model_path $MODEL_PATH \
    --port $PORT \
    #--model_path '/mnt/tjena/alex/llama2_chat`_7_31' \

