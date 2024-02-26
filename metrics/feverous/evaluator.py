import numpy as np


class EvaluateTool(object):

    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        summary = {}
        all_match = []

        stems = ["refute", "support", "not enough"]

        for pred, gold_item in zip(preds, golds):
            match_or_not = False
            for stem in stems:
                if stem in gold_item['seq_out'] and stem in pred:
                    match_or_not = True
                    break
            all_match.append(match_or_not)

        summary["all"] = float(np.mean(all_match))

        return summary
