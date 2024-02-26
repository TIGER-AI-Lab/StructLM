# encoding=utf8
import os
from .bird_evaluation import run_sqls_parallel, sort_results, compute_acc_by_diff, print_data, run
import multiprocessing as mp
import glob



class EvaluateTool(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        matching_paths = glob.glob("data/downloads/extracted/*/dev_databases")
        assert len(matching_paths) == 1
        db_base_path = matching_paths[0]
        gt_queries = [item['query'] for item in golds]
        db_paths = [os.path.join(db_base_path, item['db_id'], item['db_id'] + ".sqlite") for item in golds]
        query_pairs = list(zip(preds, gt_queries))
        exec_result = run(query_pairs, db_paths, num_cpus=4, meta_time_out=30.0)
        exec_result = sort_results(exec_result)
        
        print('start calculate')
        simple_acc, moderate_acc, challenging_acc, acc, count_lists = \
            compute_acc_by_diff(exec_result,os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev.json"))
        score_lists = [simple_acc, moderate_acc, challenging_acc, acc]
        print_data(score_lists,count_lists)
        print('===========================================================================================')
        print("Finished evaluation")

        return {"simple_acc": simple_acc, "moderate_acc": moderate_acc, "challenging_acc": challenging_acc, "acc": acc}
