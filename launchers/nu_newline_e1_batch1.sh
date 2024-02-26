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


nohup python llama_service.py --model_path="/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/models/llama/v9_nu_newline/checkpoint-2362" --port=8200 &>/dev/null &

echo "Waiting for 20 seconds as the servers start up..."
sleep 20
echo "Running inference"

./run_single_ds_eval_nl.sh nu_newline_e1 bird 1245 8200
./run_single_ds_eval_nl.sh nu_newline_e1 tabmwp 1245 8200
./run_single_ds_eval_nl.sh nu_newline_e1 logicnlg 1245 8200
./run_single_ds_eval_nl.sh nu_newline_e1 infotabs 1245 8200
./run_single_ds_eval_nl.sh nu_newline_e1 finqa 1245 8200
./run_single_ds_eval_nl.sh nu_newline_e1 wikitq 1245 8200
./run_single_ds_eval_nl.sh nu_newline_e1 hybridqa 1245 8200
./run_single_ds_eval_nl.sh nu_newline_e1 spider 1245 8200
./run_single_ds_eval_nl.sh nu_newline_e1 fetaqa 1245 8200
./run_single_ds_eval_nl.sh nu_newline_e1 sql2text 1245 8200
./run_single_ds_eval_nl.sh nu_newline_e1 dart 1245 8200
./run_single_ds_eval_nl.sh nu_newline_e1 tab_fact 1245 8200
./run_single_ds_eval_nl.sh nu_newline_e1 wikisql 1245 8200
./run_single_ds_eval_nl.sh nu_newline_e1 feverous 1245 8200
./run_single_ds_eval_nl.sh nu_newline_e1 kvret 1245 8200
./run_single_ds_eval_nl.sh nu_newline_e1 sparc 1245 8200
./run_single_ds_eval_nl.sh nu_newline_e1 cosql 1245 8200
./run_single_ds_eval_nl.sh nu_newline_e1 sqa 1245 8200
./run_single_ds_eval_nl.sh nu_newline_e1 mmqa 1245 8200
./run_single_ds_eval_nl.sh nu_newline_e1 mtop 1245 8200
./run_single_ds_eval_nl.sh nu_newline_e1 logic2text 1245 8200
./run_single_ds_eval_nl.sh nu_newline_e1 totto 1245 8200
