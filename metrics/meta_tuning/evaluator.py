'''
Author: ygjin11 1633504509@qq.com
Date: 2024-02-15 04:38:45
LastEditors: ygjin11 1633504509@qq.com
LastEditTime: 2024-02-15 04:38:45
FilePath: /uskg_eval/metrics/meta_tuning/evaluator.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf8
import os
import copy
import numpy as np

import utils.tool
from utils.configue import Configure
from tqdm import tqdm


class EvaluateTool(object):
    """
    The meta evaluator
    """
    def __init__(self, meta_args):
        self.meta_args = meta_args

    def evaluate(self, preds, golds, section):
        meta_args = self.meta_args
        summary = {}
        wait_for_eval = {}

        for pred, gold in zip(preds, golds):
            if gold['arg_path'] not in wait_for_eval.keys():
                wait_for_eval[gold['arg_path']] = {'preds': [], "golds":[]}
            wait_for_eval[gold['arg_path']]['preds'].append(pred)
            wait_for_eval[gold['arg_path']]['golds'].append(gold)

        lst = [(arg_path, preds_golds) for arg_path, preds_golds in wait_for_eval.items()]
        print([arg_path for arg_path, preds_golds in lst])
        for arg_path, preds_golds in tqdm(lst):
            # if "tabmwp" not in arg_path:
            #     continue
            print("Evaluating {}...".format(arg_path))
            args = Configure.refresh_args_by_file_cfg(os.path.join(meta_args.dir.configure, arg_path), meta_args)
            evaluator = utils.tool.get_evaluator(args.evaluate.tool)(args)
            summary_tmp = evaluator.evaluate(preds_golds['preds'], preds_golds['golds'], section)
            print(summary_tmp)
            for key, metric in summary_tmp.items():  # TODO
                summary[os.path.join(arg_path, key)] = metric
            # summary[os.path.join(arg_path, args.train.stop)] = summary_tmp[args.train.stop]

        summary['avr'] = float(np.mean([float(v) for k, v in summary.items()]))
        return summary
