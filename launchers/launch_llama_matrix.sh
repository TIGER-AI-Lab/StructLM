# we want to launch 4 processes for each llama version of 
# /cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/models/llama/v7_non_upsampled/checkpoint-2362
# /cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/models/llama/v7_non_upsampled/checkpoint-4724
# /cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/models/llama/v7_non_upsampled/checkpoint-7086

# for the first checkpoint

#CKPT_BASE=/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/models/llama/v7_non_upsampled
CKPT_BASE=/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/models/llama/v6_upsampled_rcs
PORT=8200
# we will use ports 8080-8092

# make a list of checkpoints 
# checkpoints is an array of [checkpoint-2362, checkpoint-4724, checkpoint-7086]
#checkpoints=("checkpoint-2362" "checkpoint-4724" "checkpoint-7086")
checkpoints=("checkpoint-7776" "checkpoint-11664")
for i in {0..1}; do
    for checkpoint in "${checkpoints[@]}"; do
    #echo "launching llama servers for $CKPT_PATH"
    # run the llama_service.py script in the background
        CKPT_PATH=$CKPT_BASE/$checkpoint
        #echo "launching llama server on port $PORT"
        echo "export CUDA_VISIBLE_DEVICES=$i && python llama_service.py --model_path $CKPT_PATH --port $PORT"
        # increment the port number
        PORT=$((PORT+1))
        # wait for 10 seconds
        #sleep 10
    done
    
done

# python llama_service.py --model_path /cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/models/llama/v7_non_upsampled/checkpoint-2362 --port 8100
# python llama_service.py --model_path /cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/models/llama/v7_non_upsampled/checkpoint-4724 --port 8104
# python llama_service.py --model_path /cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/models/llama/v7_non_upsampled/checkpoint-7086 --port 8108