#!/bin/bash

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


nohup python llama_service.py --model_path="/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/models/llama/v9_nu_newline/checkpoint-4724" --port=8202 &>/dev/null &

echo "Waiting for 20 seconds as the servers start up..."
sleep 20
echo "Running inference"

./run_single_ds_eval_nl.sh nu_newline_e2 bird 1247 8202
./run_single_ds_eval_nl.sh nu_newline_e2 tabmwp 1247 8202
./run_single_ds_eval_nl.sh nu_newline_e2 logicnlg 1247 8202
./run_single_ds_eval_nl.sh nu_newline_e2 infotabs 1247 8202
./run_single_ds_eval_nl.sh nu_newline_e2 finqa 1247 8202
./run_single_ds_eval_nl.sh nu_newline_e2 wikitq 1247 8202
./run_single_ds_eval_nl.sh nu_newline_e2 hybridqa 1247 8202
./run_single_ds_eval_nl.sh nu_newline_e2 spider 1247 8202
./run_single_ds_eval_nl.sh nu_newline_e2 fetaqa 1247 8202
./run_single_ds_eval_nl.sh nu_newline_e2 sql2text 1247 8202
./run_single_ds_eval_nl.sh nu_newline_e2 dart 1247 8202
./run_single_ds_eval_nl.sh nu_newline_e2 tab_fact 1247 8202
./run_single_ds_eval_nl.sh nu_newline_e2 wikisql 1247 8202
./run_single_ds_eval_nl.sh nu_newline_e2 feverous 1247 8202
./run_single_ds_eval_nl.sh nu_newline_e2 kvret 1247 8202
./run_single_ds_eval_nl.sh nu_newline_e2 sparc 1247 8202
./run_single_ds_eval_nl.sh nu_newline_e2 cosql 1247 8202
./run_single_ds_eval_nl.sh nu_newline_e2 sqa 1247 8202
./run_single_ds_eval_nl.sh nu_newline_e2 mmqa 1247 8202
./run_single_ds_eval_nl.sh nu_newline_e2 mtop 1247 8202
./run_single_ds_eval_nl.sh nu_newline_e2 logic2text 1247 8202
./run_single_ds_eval_nl.sh nu_newline_e2 totto 1247 8202
