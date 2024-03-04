import logging
import os
import time

import torch
import importlib
import datasets
import transformers
from transformers import (
    HfArgumentParser,
    set_seed,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from collections import OrderedDict
import utils.tool
from utils.configure import Configure
from utils.dataset import TokenizedTestDataset
from utils.trainer import LlamaSeq2SeqTrainer
from utils.training_arguments import WrappedSeq2SeqTrainingArguments
import json
from vllm import LLM

# Huggingface realized the "Seq2seqTrainingArguments" which is the same with "WrappedSeq2SeqTrainingArguments"
# in transformers==4.10.1 during our work.
logger = logging.getLogger(__name__)

# class with a getitem
class DummyDataset():
    def __getitem__(self, index):
        return {}

def get_model(model):
    Model = importlib.import_module('models.{}'.format(model)).Model
    return Model

def get_evaluator(evaluate_tool):
    EvaluateTool = importlib.import_module('{}'.format(evaluate_tool)).EvaluateTool
    return EvaluateTool

def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # Get args
    parser = HfArgumentParser((WrappedSeq2SeqTrainingArguments,))
    training_args, = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    args = Configure.Get(training_args.cfg)

    evaluator = utils.tool.get_evaluator(args.evaluate.tool)(args)
    model = utils.tool.get_model(args.model.name)(args)
    model_tokenizer = model.tokenizer

    logging.info(f"loading test data from file {args.dataset.test_split_json}")
    assert args.dataset.test_split_json is not None, "Please specify the test split json file."
    with open(args.dataset.test_split_json) as f:
        seq2seq_test_dataset= json.load(f)

    test_dataset = TokenizedTestDataset(args, training_args, model_tokenizer,
                                    seq2seq_test_dataset) if seq2seq_test_dataset else None

    # Initialize our Trainer
    trainer = LlamaSeq2SeqTrainer(
        args=training_args,
        model=model,
        evaluator=evaluator,
        tokenizer=model_tokenizer,
    )
    logging.info('Trainer build successfully.')


    logger.info("*** Predict ***")

    predict_results = trainer.predict(
        test_dataset=test_dataset,
        test_examples=seq2seq_test_dataset,
        metric_key_prefix="predict"
    )

if __name__ == "__main__":
    main()