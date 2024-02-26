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

./run_t5_single_eval.sh t5_large_baseline bird 1235 
./run_t5_single_eval.sh t5_large_baseline infotabs 1235 
./run_t5_single_eval.sh t5_large_baseline finqa 1235 
./run_t5_single_eval.sh t5_large_baseline tabmwp 1235 
./run_t5_single_eval.sh t5_large_baseline wikitq 1235 
./run_t5_single_eval.sh t5_large_baseline hybridqa 1235 
./run_t5_single_eval.sh t5_large_baseline spider 1235 
./run_t5_single_eval.sh t5_large_baseline fetaqa 1235 
./run_t5_single_eval.sh t5_large_baseline sql2text 1235 
./run_t5_single_eval.sh t5_large_baseline dart 1235 
./run_t5_single_eval.sh t5_large_baseline tab_fact 1235 
./run_t5_single_eval.sh t5_large_baseline wikisql 1235 
./run_t5_single_eval.sh t5_large_baseline feverous 1235 
./run_t5_single_eval.sh t5_large_baseline kvret 1235 
./run_t5_single_eval.sh t5_large_baseline sparc 1235 
./run_t5_single_eval.sh t5_large_baseline cosql 1235 
./run_t5_single_eval.sh t5_large_baseline sqa 1235 
./run_t5_single_eval.sh t5_large_baseline mtop 1235 
./run_t5_single_eval.sh t5_large_baseline logic2text 1235 
./run_t5_single_eval.sh t5_large_baseline totto 1235 
./run_t5_single_eval.sh t5_large_baseline multiwoz 1235 