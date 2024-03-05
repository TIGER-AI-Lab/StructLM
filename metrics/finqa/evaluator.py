# encoding=utf8
import numpy as np
from metrics.finqa.finqa_evaluation import eval_program, program_tokenization, equal_program

class EvaluateTool(object):

    def __init__(self, args):
        self.args = args

    def python_expression_eval(self, preds, golds):
        total = 0
        correct = 0
        for pred, gold_item in zip(preds, golds):
            # fall back to evaluating the python expression.
            if pred.lower().endswith(gold_item['final_res'].lower()):
                # for non numeric answers, just check if the answer is in the prediction
                correct += 1
            else:
                # first remove all percent signs and money signs from the answer
                pred = pred.replace('%', '').replace("$", '')
                # if it contains an equal sign, take the part before the equal sign
                if '=' in pred:
                    pred = pred.split('=')[0]

                # if gold is a percentage, remove the percent sign and express as a decimal
                if gold_item['final_res'].endswith('%'):
                    gold = float(gold_item['final_res'].replace('%', '')) / 100
                # try to evaluate the expression
                else:
                    try:
                        # not a percentage, and can't be converted to a float
                        gold = float(eval(gold_item['final_res']))
                    except:
                        pass
                try:
                    pred = float(eval(pred))
                    # round to the same number of decimal places as the gold answer
                    pred = round(pred, len(str(gold).split('.')[1]))
                    # if the prediction is close enough to the gold answer, count as correct
                    if np.isclose(pred, gold, atol=0.001):
                        correct += 1
                except:
                    # count as incorrect
                    pass
            total += 1
        return float(correct) / total

    def finqa_eval(self, preds, golds):
        assert len(preds) == len(golds)
        exe_correct = 0
        prog_correct = 0

        res_list = []
        all_res_list = []
        
        for pred, gold_item in zip(preds, golds):
            pred = program_tokenization(pred)
            each_id = gold_item["id"]
            table = gold_item["table"]
            gold_res = gold_item["qa"]["exe_ans"]
            gold = program_tokenization(gold_item["qa"]["program"])
            invalid_flag, exe_res = eval_program(pred, table)       
            if invalid_flag == 0:
                if exe_res == gold_res:
                    exe_correct += 1                    
            if equal_program(gold, pred):
                # assert exe_res == gold_res
                if exe_res != gold_res:
                    print(each_id)
                    print(gold)
                    print(pred)
                    print(gold_res)
                    print(exe_res)
                    print(gold_item["id"])
                assert exe_res == gold_res
                prog_correct += 1
                if "".join(gold) != "".join(pred):
                    print(each_id)
                    print(gold)
                    print(pred)
                    print(gold_res)
                    print(exe_res)
                    print(gold_item["id"])

            gold_item["qa"]["predicted"] = pred

            if exe_res != gold_res:
                res_list.append(gold_item)
            all_res_list.append(gold_item)
                
        exe_acc = float(exe_correct) / len(pred)
        prog_acc = float(prog_correct) / len(pred)
                
        # print("All: ", len(data))
        # print("Exe acc: ", exe_acc)
        # print("Prog acc: ", prog_acc)
        return exe_acc


    def evaluate(self, preds, golds, section):

        acc = 0
        try:
            # try to evaluate the predictions as finqa expressions
            acc = self.finqa_eval(preds, golds)
            print("Acc: ", acc)
        except:
            # fall back to evaluating the python expression.
            acc = max(self.python_expression_eval(preds, golds), acc)
        return {"acc": acc}
        
