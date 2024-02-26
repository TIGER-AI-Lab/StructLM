split -l 13 single_task_eval.sh single_task_eval_ # initial split
i=1
for file in single_task_eval_*; do
	  mv "$file" "single_task_eval_${i}.sh"
	    ((i++))
    done
