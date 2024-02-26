export CUDA_VISIBLE_DEVICES=1
export WANDB_MODE=disabled

kwargs=" --logging_strategy steps \
--do_predict \
--predict_with_generate \
--overwrite_output_dir \
--per_device_eval_batch_size 1 \
--generation_max_length 256 \
--input_max_length 1792 \
--ddp_find_unused_parameters true \
"

CFG_PREFIX="llamax_e2"

# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_compwebq --output_dir output/${CFG_PREFIX}_compwebq --cfg Salesforce/${CFG_PREFIX}_compwebq.cfg
# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_cosql --output_dir output/${CFG_PREFIX}_cosql --cfg Salesforce/${CFG_PREFIX}_cosql.cfg
# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_dart --output_dir output/${CFG_PREFIX}_dart --cfg Salesforce/${CFG_PREFIX}_dart.cfg
# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_fetaqa --output_dir output/${CFG_PREFIX}_fetaqa --cfg Salesforce/${CFG_PREFIX}_fetaqa.cfg
# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_feverous --output_dir output/${CFG_PREFIX}_feverous --cfg Salesforce/${CFG_PREFIX}_feverous.cfg
# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_grailqa --output_dir output/${CFG_PREFIX}_grailqa --cfg Salesforce/${CFG_PREFIX}_grailqa.cfg
# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_hybridqa --output_dir output/${CFG_PREFIX}_hybridqa --cfg Salesforce/${CFG_PREFIX}_hybridqa.cfg
# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_kvret --output_dir output/${CFG_PREFIX}_kvret --cfg Salesforce/${CFG_PREFIX}_kvret.cfg
# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_logic2text --output_dir output/${CFG_PREFIX}_logic2text --cfg Salesforce/${CFG_PREFIX}_logic2text.cfg
# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_mmqa --output_dir output/${CFG_PREFIX}_mmqa --cfg Salesforce/${CFG_PREFIX}_mmqa.cfg
# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_mtop --output_dir output/${CFG_PREFIX}_mtop --cfg Salesforce/${CFG_PREFIX}_mtop.cfg
# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_multiwoz --output_dir output/${CFG_PREFIX}_multiwoz --cfg Salesforce/${CFG_PREFIX}_multiwoz.cfg
# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_spider --output_dir output/${CFG_PREFIX}_spider --cfg Salesforce/${CFG_PREFIX}_spider.cfg
# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_sparc --output_dir output/${CFG_PREFIX}_sparc --cfg Salesforce/${CFG_PREFIX}_sparc.cfg
# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_sqa --output_dir output/${CFG_PREFIX}_sqa --cfg Salesforce/${CFG_PREFIX}_sqa.cfg
# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_sql2text --output_dir output/${CFG_PREFIX}_sql2text --cfg Salesforce/${CFG_PREFIX}_sql2text.cfg
# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_tab_fact --output_dir output/${CFG_PREFIX}_tab_fact --cfg Salesforce/${CFG_PREFIX}_tab_fact.cfg
# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_totto --output_dir output/${CFG_PREFIX}_totto --cfg Salesforce/${CFG_PREFIX}_totto.cfg
# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_webqsp --output_dir output/${CFG_PREFIX}_webqsp --cfg Salesforce/${CFG_PREFIX}_webqsp.cfg
# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_wikisql --output_dir output/${CFG_PREFIX}_wikisql --cfg Salesforce/${CFG_PREFIX}_wikisql.cfg
# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_wikitq --output_dir output/${CFG_PREFIX}_wikitq --cfg Salesforce/${CFG_PREFIX}_wikitq.cfg


# # mod = 1
torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_totto --output_dir output/${CFG_PREFIX}_totto --cfg Salesforce/${CFG_PREFIX}_totto.cfg
torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_wikitq --output_dir output/${CFG_PREFIX}_wikitq --cfg Salesforce/${CFG_PREFIX}_wikitq.cfg
torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_hybridqa --output_dir output/${CFG_PREFIX}_hybridqa --cfg Salesforce/${CFG_PREFIX}_hybridqa.cfg
torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_spider --output_dir output/${CFG_PREFIX}_spider --cfg Salesforce/${CFG_PREFIX}_spider.cfg
torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_compwebq --output_dir output/${CFG_PREFIX}_compwebq --cfg Salesforce/${CFG_PREFIX}_compwebq.cfg
torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_fetaqa --output_dir output/${CFG_PREFIX}_fetaqa --cfg Salesforce/${CFG_PREFIX}_fetaqa.cfg
torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_mmqa --output_dir output/${CFG_PREFIX}_mmqa --cfg Salesforce/${CFG_PREFIX}_mmqa.cfg
torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_sql2text --output_dir output/${CFG_PREFIX}_sql2text --cfg Salesforce/${CFG_PREFIX}_sql2text.cfg
torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_dart --output_dir output/${CFG_PREFIX}_dart --cfg Salesforce/${CFG_PREFIX}_dart.cfg

# mod = 2
# torchrun --nproc_per_node 1 --master_port=1236 eval_llama.py $kwargs --run_name llama2_cosql --output_dir output/${CFG_PREFIX}_cosql --cfg Salesforce/${CFG_PREFIX}_cosql.cfg
# torchrun --nproc_per_node 1 --master_port=1236 eval_llama.py $kwargs --run_name llama2_sparc --output_dir output/${CFG_PREFIX}_sparc --cfg Salesforce/${CFG_PREFIX}_sparc.cfg
# torchrun --nproc_per_node 1 --master_port=1236 eval_llama.py $kwargs --run_name llama2_tab_fact --output_dir output/${CFG_PREFIX}_tab_fact --cfg Salesforce/${CFG_PREFIX}_tab_fact.cfg
# torchrun --nproc_per_node 1 --master_port=1236 eval_llama.py $kwargs --run_name llama2_wikisql --output_dir output/${CFG_PREFIX}_wikisql --cfg Salesforce/${CFG_PREFIX}_wikisql.cfg
# torchrun --nproc_per_node 1 --master_port=1236 eval_llama.py $kwargs --run_name llama2_feverous --output_dir output/${CFG_PREFIX}_feverous --cfg Salesforce/${CFG_PREFIX}_feverous.cfg
# torchrun --nproc_per_node 1 --master_port=1236 eval_llama.py $kwargs --run_name llama2_kvret --output_dir output/${CFG_PREFIX}_kvret --cfg Salesforce/${CFG_PREFIX}_kvret.cfg
# torchrun --nproc_per_node 1 --master_port=1236 eval_llama.py $kwargs --run_name llama2_mtop --output_dir output/${CFG_PREFIX}_mtop --cfg Salesforce/${CFG_PREFIX}_mtop.cfg

# # mod = 3  
# torchrun --nproc_per_node 1 --master`_port=1237 eval_llama.py $kwargs --run_name llama2_sqa --output_dir output/${CFG_PREFIX}_sqa --cfg Salesforce/${CFG_PREFIX}_sqa.cfg
# torchrun --nproc_per_node 1 --master_port=1237 eval_llama.py $kwargs --run_name llama2_logic2text --output_dir output/${CFG_PREFIX}_logic2text --cfg Salesforce/${CFG_PREFIX}_logic2text.cfg
# torchrun --nproc_per_node 1 --master_port=1237 eval_llama.py $kwargs --run_name llama2_multiwoz --output_dir output/${CFG_PREFIX}_multiwoz --cfg Salesforce/${CFG_PREFIX}_multiwoz.cfg
# torchrun --nproc_per_node 1 --master_port=1237 eval_llama.py $kwargs --run_name llama2_totto --output_dir output/${CFG_PREFIX}_totto --cfg Salesforce/${CFG_PREFIX}_totto.cfg
# torchrun --nproc_per_node 1 --master_port=1237 eval_llama.py $kwargs --run_name llama2_wikitq --output_dir output/${CFG_PREFIX}_wikitq --cfg Salesforce/${CFG_PREFIX}_wikitq.cfg
# torchrun --nproc_per_node 1 --master_port=1235 eval_llama.py $kwargs --run_name llama2_webqsp --output_dir output/${CFG_PREFIX}_webqsp --cfg Salesforce/${CFG_PREFIX}_webqsp.cfg
# torchrun --nproc_per_node 1 --master_port=1237 eval_llama.py $kwargs --run_name llama2_grailqa --output_dir output/${CFG_PREFIX}_grailqa --cfg Salesforce/${CFG_PREFIX}_grailqa.cfg
