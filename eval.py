import logging
import os
import time

import torch
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
from utils.configue import Configure
from utils.dataset import TokenizedDataset, TokenizedLlamaDataset, TokenizedTestDataset
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

def main() -> None:
    os.environ[
        'CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # Deterministic behavior of torch.addmm. Please refer to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    #torch.use_deterministic_algorithms(True) # NOTE this causes issues with VLLM
    # Initialize the logger
    logging.basicConfig(level=logging.INFO)

    # Get args
    parser = HfArgumentParser((WrappedSeq2SeqTrainingArguments,))
    training_args, = parser.parse_args_into_dataclasses()
    set_seed(training_args.seed)
    args = Configure.Get(training_args.cfg)

    evaluator = utils.tool.get_evaluator(args.evaluate.tool)(args)
    model = utils.tool.get_model(args.model.name)(args)
    model_tokenizer = model.tokenizer
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     args.model.path,``
    #     torch_dtype=torch.float16,
    # )
    

    logging.info(f"loading test data from file {args.dataset.test_split_json}")
    assert args.dataset.test_split_json is not None, "Please specify the test split json file."
    with open(args.dataset.test_split_json) as f:
        seq2seq_test_dataset= json.load(f)

    # # NOTE remove this after testing complete.
    # dnames = set([each['arg_path'] for each in seq2seq_test_dataset])
    # # select only one example for each dname
    # small_dataset = []
    # for i, each in enumerate(seq2seq_test_dataset):
    #     if each['arg_path'] in dnames:
    #         dnames.remove(each['arg_path'])
    #         small_dataset.append(each)
    # seq2seq_test_dataset = small_dataset
    # import pdb; pdb.set_trace()
    # We wrap the "string" seq2seq data into "tokenized tensor".
    test_dataset = TokenizedTestDataset(args, training_args, model_tokenizer,
                                    seq2seq_test_dataset) if seq2seq_test_dataset else None

    # Initialize our Trainer
    trainer = LlamaSeq2SeqTrainer(
        args=training_args,
        model=model,
        evaluator=evaluator,
        # We name it "evaluator" while the hugging face call it "Metric",
        # they are all f(predictions: List, references: List of dict) = eval_result: dict
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