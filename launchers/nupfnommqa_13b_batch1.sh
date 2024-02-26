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


nohup python llama_service.py --model_path="/cpfs/29cd2992fe666f2a/user/huangwenhao/alex/uskg_eval/models/llama/v10_nu_prompt_fixed_13b/checkpoint-2301" --port=8200 &>/dev/null &

echo "Waiting for 120 seconds as the servers start up..."
sleep 120
echo "Running inference"

./run_single_ds_eval.sh nu_pf_nommqa_13b bird 1245 8200
./run_single_ds_eval.sh nu_pf_nommqa_13b tabmwp 1245 8200
#./run_single_ds_eval.sh nu_pf_nommqa_13b logicnlg 1245 8200
./run_single_ds_eval.sh nu_pf_nommqa_13b infotabs 1245 8200
./run_single_ds_eval.sh nu_pf_nommqa_13b finqa 1245 8200
./run_single_ds_eval.sh nu_pf_nommqa_13b wikitq 1245 8200
./run_single_ds_eval.sh nu_pf_nommqa_13b hybridqa 1245 8200
./run_single_ds_eval.sh nu_pf_nommqa_13b spider 1245 8200
./run_single_ds_eval.sh nu_pf_nommqa_13b fetaqa 1245 8200
./run_single_ds_eval.sh nu_pf_nommqa_13b sql2text 1245 8200
./run_single_ds_eval.sh nu_pf_nommqa_13b dart 1245 8200
./run_single_ds_eval.sh nu_pf_nommqa_13b tab_fact 1245 8200
./run_single_ds_eval.sh nu_pf_nommqa_13b wikisql 1245 8200
./run_single_ds_eval.sh nu_pf_nommqa_13b feverous 1245 8200
./run_single_ds_eval.sh nu_pf_nommqa_13b kvret 1245 8200
./run_single_ds_eval.sh nu_pf_nommqa_13b sparc 1245 8200
./run_single_ds_eval.sh nu_pf_nommqa_13b cosql 1245 8200
./run_single_ds_eval.sh nu_pf_nommqa_13b sqa 1245 8200
# ./run_single_ds_eval.sh nu_pf_nommqa_13b mmqa 1245 8200
./run_single_ds_eval.sh nu_pf_nommqa_13b mtop 1245 8200
./run_single_ds_eval.sh nu_pf_nommqa_13b logic2text 1245 8200
./run_single_ds_eval.sh nu_pf_nommqa_13b totto 1245 8200
