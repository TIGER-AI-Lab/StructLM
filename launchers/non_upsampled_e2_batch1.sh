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
conda activate SKGLM


nohup python llama_service.py --model_path="/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/models/llama/v7_non_upsampled/checkpoint-4724" --port=8100 &>/dev/null &

echo "Waiting for 20 seconds as the servers start up..."
sleep 20
echo "Running inference"

# bird, infotabs, finqa, logicnlg, tabmwp
./run_single_ds_eval.sh non_upsampled_e2 bird 1235 8100
./run_single_ds_eval.sh non_upsampled_e2 infotabs 1235 8100
./run_single_ds_eval.sh non_upsampled_e2 finqa 1235 8100
./run_single_ds_eval.sh non_upsampled_e2 logicnlg 1235 8100
./run_single_ds_eval.sh non_upsampled_e2 tabmwp 1235 8100
./run_single_ds_eval.sh non_upsampled_e2 wikitq 1235 8100
./run_single_ds_eval.sh non_upsampled_e2 hybridqa 1235 8100
./run_single_ds_eval.sh non_upsampled_e2 spider 1235 8100
./run_single_ds_eval.sh non_upsampled_e2 fetaqa 1235 8100
./run_single_ds_eval.sh non_upsampled_e2 sql2text 1235 8100
./run_single_ds_eval.sh non_upsampled_e2 dart 1235 8100
./run_single_ds_eval.sh non_upsampled_e2 tab_fact 1235 8100
./run_single_ds_eval.sh non_upsampled_e2 wikisql 1235 8100
./run_single_ds_eval.sh non_upsampled_e2 feverous 1235 8100
./run_single_ds_eval.sh non_upsampled_e2 kvret 1235 8100
./run_single_ds_eval.sh non_upsampled_e2 sparc 1235 8100
./run_single_ds_eval.sh non_upsampled_e2 cosql 1235 8100
./run_single_ds_eval.sh non_upsampled_e2 sqa 1235 8100
./run_single_ds_eval.sh non_upsampled_e2 mmqa 1235 8100
./run_single_ds_eval.sh non_upsampled_e2 mtop 1235 8100
./run_single_ds_eval.sh non_upsampled_e2 logic2text 1235 8100
./run_single_ds_eval.sh non_upsampled_e2 totto 1235 8100
