# encoding=utf8
import numpy as np
from datasets import load_metric


#  the code below refers to the https://github.com/Yale-LILY/FeTaQA/blob/main/end2end/train.py
def postprocess_text(preds, references_s, metric_name):
    preds = [pred.strip() for pred in preds]
    references_s = [[reference.strip() for reference in references] for references in references_s]

    # rougeLSum expects newline after each sentence
    if metric_name in ["sacrebleu"]:
        # since hf sacrebleu only support references with same length, we have to pad them into the same length
        ref_max_len = max([len(ref) for ref in references_s])
        # see https://github.com/mjpost/sacrebleu/pull/132
        references_s = [references + [None] * (ref_max_len - len(references)) for references in references_s]
    else:
        pass

    return preds, references_s


class EvaluateTool(object):
    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        summary = {}
        references_s = [[item["seq_out"]] for item in golds]

        assert len(preds) == len(references_s)

        metric_list = []
        if section in ['train', 'dev']:
            metric_list = ['sacrebleu']
        elif section == 'test':
            metric_list = ["sacrebleu"]#, "bleurt"]  # TODO: add PARENT

        for metric_name in metric_list:
            metric = load_metric(metric_name)
            processed_preds, processed_golds = postprocess_text(preds, references_s, metric_name)
            if metric_name == "sacrebleu":
                res = metric.compute(predictions=processed_preds, references=processed_golds)
                summary[metric_name] = res["score"] * 0.01
                print(metric_name, res)
            else:
                res = metric.compute(predictions=processed_preds, references=processed_golds)
                summary[metric_name] = res[metric_name]
        return summary