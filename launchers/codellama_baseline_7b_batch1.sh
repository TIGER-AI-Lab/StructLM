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


nohup python llama_service.py --model_path="/cpfs/29cd2992fe666f2a/shared/public/hub/models--codellama--CodeLlama-7b-Instruct-hf/snapshots/6114dd1e16f69e0765ccbd7a64d33d04b265fbd2" --port=8200 &>/dev/null &

echo "Waiting for 20 seconds as the servers start up..."
sleep 20
echo "Running inference"

./run_single_ds_eval.sh codellama_baseline_7b bird 1245 8200
./run_single_ds_eval.sh codellama_baseline_7b infotabs 1245 8200
./run_single_ds_eval.sh codellama_baseline_7b cosql 1245 8200
./run_single_ds_eval.sh codellama_baseline_7b sqa 1245 8200
./run_single_ds_eval.sh codellama_baseline_7b finqa 1245 8200

./run_single_ds_eval.sh codellama_baseline_7b tabmwp 1245 8200
./run_single_ds_eval.sh codellama_baseline_7b wikitq 1245 8200
./run_single_ds_eval.sh codellama_baseline_7b tab_fact 1245 8200
./run_single_ds_eval.sh codellama_baseline_7b hybridqa 1245 8200
./run_single_ds_eval.sh codellama_baseline_7b spider 1245 8200
./run_single_ds_eval.sh codellama_baseline_7b fetaqa 1245 8200
./run_single_ds_eval.sh codellama_baseline_7b sql2text 1245 8200
./run_single_ds_eval.sh codellama_baseline_7b dart 1245 8200
./run_single_ds_eval.sh codellama_baseline_7b totto 1245 8200
./run_single_ds_eval.sh codellama_baseline_7b logic2text 1245 8200
./run_single_ds_eval.sh codellama_baseline_7b mtop 1245 8200
# ./run_single_ds_eval.sh codellama_baseline_7b logicnlg 1245 8200
#./run_single_ds_eval.sh codellama_baseline_7b wikisql 1245 8200
#./run_single_ds_eval.sh codellama_baseline_7b feverous 1245 8200
#./run_single_ds_eval.sh codellama_baseline_7b kvret 1245 8200
#./run_single_ds_eval.sh codellama_baseline_7b sparc 1245 8200
#./run_single_ds_eval.sh codellama_baseline_7b mmqa 1245 8200
