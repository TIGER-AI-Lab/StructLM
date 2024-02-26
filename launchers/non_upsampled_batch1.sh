# #!/bin/bash
# __conda_setup="$('/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/miniconda3/etc/profile.d/conda.sh" ]; then
#         . "/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/miniconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/miniconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup

# which conda
# export HF_HOME=/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/.cache/huggingface
# cd /cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/
# conda activate SKGLM


# nohup python llama_service.py --model_path="/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/models/llama/v7_non_upsampled/checkpoint-2362" --port=8100 &>/dev/null &

# echo "Waiting for 20 seconds as the servers start up..."
# sleep 20
# echo "Running inference"

./run_single_ds_eval.sh non_upsampled bird 1235 8100
./run_single_ds_eval.sh non_upsampled infotabs 1235 8100
./run_single_ds_eval.sh non_upsampled finqa 1235 8100
./run_single_ds_eval.sh non_upsampled logicnlg 1235 8100
./run_single_ds_eval.sh non_upsampled tabmwp 1235 8100
./run_single_ds_eval.sh non_upsampled wikitq 1235 8100
./run_single_ds_eval.sh non_upsampled hybridqa 1235 8100
./run_single_ds_eval.sh non_upsampled spider 1235 8100
./run_single_ds_eval.sh non_upsampled fetaqa 1235 8100
./run_single_ds_eval.sh non_upsampled sql2text 1235 8100
./run_single_ds_eval.sh non_upsampled dart 1235 8100
./run_single_ds_eval.sh non_upsampled tab_fact 1235 8100
./run_single_ds_eval.sh non_upsampled wikisql 1235 8100
./run_single_ds_eval.sh non_upsampled feverous 1235 8100
./run_single_ds_eval.sh non_upsampled kvret 1235 8100
./run_single_ds_eval.sh non_upsampled sparc 1235 8100
./run_single_ds_eval.sh non_upsampled cosql 1235 8100
./run_single_ds_eval.sh non_upsampled sqa 1235 8100
./run_single_ds_eval.sh non_upsampled mmqa 1235 8100
./run_single_ds_eval.sh non_upsampled mtop 1235 8100
./run_single_ds_eval.sh non_upsampled logic2text 1235 8100
./run_single_ds_eval.sh non_upsampled totto 1235 8100
