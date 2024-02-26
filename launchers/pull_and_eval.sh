#"wikitq", "hybridqa", "spider", "fetaqa", "sql2text", "dart", "tab_fact", "wikisql", "feverous", "kvret", "sparc", "cosql","sqa", "mmqa", "mtop", "logic2text","totto", "multiwoz"
DATASET=("finqa" "logicnlg" "bird" "infotabs" "tabmwp" "wikitq" "hybridqa" "spider" "fetaqa" "sql2text" "tab_fact" "wikisql" "feverous" "kvret" "sparc" "cosql" "sqa" "mmqa" "mtop" "logic2text" "multiwoz" "totto" "dart")

RUN_NAME=$1

for dataset_name in "${DATASET[@]}"; do
    ./pull_single_ds_and_eval.sh $RUN_NAME $dataset_name
done