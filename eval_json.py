import argparse
import importlib
import os
import json
from utils.configure import Configure

def eval_loose_json(args):
    cfgargs = Configure.Get('')
    evaluator = importlib.import_module("metrics.meta_tuning.evaluator").EvaluateTool(cfgargs)
    with open(args.json_file, "rb") as f:
        data = json.load(f)
    preds = [item['prediction'] for item in data]
    labs = data
    summary = evaluator.evaluate(preds, labs, "test")
    print(summary)

def main(args):
    # use import lib to import EvaluateTool from metrics.{args.dataset_name}.evaluator
    output_path= f"./output/{args.run_name}"
    predictions_path = os.path.join(output_path,"predictions_predict.json")
    config_path = f"{args.run_name}.cfg"
    args = Configure.Get(config_path)
    evaluator = importlib.import_module("metrics.meta_tuning.evaluator").EvaluateTool(args)


    with open(predictions_path, "rb") as f:
        data = json.load(f)
    preds = [item['prediction'] for item in data]
    labs = data
    summary = evaluator.evaluate(preds, labs, "test")
    print(summary)
    with open(os.path.join(output_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

if __name__=="__main__":
    # args: name of the data, name of the run, 
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="run1")
    parser.add_argument("--json_file", type=str, default=None)
    args = parser.parse_args()
    if args.json_file:
        eval_loose_json(args)
    else:
        main(args)