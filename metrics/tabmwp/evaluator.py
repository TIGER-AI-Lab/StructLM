# encoding=utf8
import numpy as np
from .tabmwp_evaluate_acc import normalize_answer, extract_prediction

class EvaluateTool(object):

    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):

        total = 0
        correct = 0
        for pred, gold_item in zip(preds, golds): 
            output = pred

            options = None
            if gold_item['choices']:
                options = eval(gold_item['choices'])

            # extract theprediction answer
            prediction = extract_prediction(output, options)

            answer_norm = normalize_answer(gold_item['answer'], gold_item['unit'])
            prediction_norm = normalize_answer(prediction, gold_item['unit'])
            # correct or not
            if answer_norm.lower() == prediction_norm.lower():
                correct += 1
            total += 1
        print("Acc: ", correct/total)
        return {"acc": correct/total}
