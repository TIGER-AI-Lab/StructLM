import collections
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from typing import NamedTuple

import datasets
import numpy as np
import torch
import transformers.trainer_seq2seq
from torch.utils.data import Dataset
from packaging import version
from torch import nn
from transformers.trainer_utils import PredictionOutput, speed_metrics

from .training_arguments import WrappedSeq2SeqTrainingArguments

_is_torch_generator_available = False
if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_torch_generator_available = True
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

from vllm import SamplingParams

class EvalPrediction(NamedTuple):
    predictions: List[str]
    items: List[dict]

class LlamaSeq2SeqTrainer(transformers.trainer_seq2seq.Seq2SeqTrainer):
    def __init__(
            self,
            evaluator,
            *args: WrappedSeq2SeqTrainingArguments,
            eval_examples: Optional[Dataset] = None,
            ignore_pad_token_for_loss: bool = True,
            wandb_run_dir: Optional[str] = None,
            **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.evaluator = evaluator
        self.eval_examples = eval_examples
        self.compute_metrics = self._compute_metrics
        self.ignore_pad_token_for_loss = ignore_pad_token_for_loss
        self.wandb_run_dir = wandb_run_dir


    def predict(
            self,
            test_dataset: Optional[Dataset],
            test_examples: Optional[Dataset],
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
            max_length: Optional[int] = None,
            max_time: Optional[int] = None,
            num_beams: Optional[int] = None,
    ) -> PredictionOutput:
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams
        self._max_time = max_time

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.evaluation_loop(
                test_dataloader,
                description="Prediction",
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.compute_metrics is not None:

            eval_preds = self._post_process_function(
                test_examples, output.predictions, metric_key_prefix)
            output.metrics.update(self.compute_metrics(eval_preds, section="test"))

        output.metrics.update(speed_metrics(metric_key_prefix, start_time, len(test_dataset)))

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(output.metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                output.metrics[f"{metric_key_prefix}_{key}"] = output.metrics.pop(key)

        self.log(output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output

    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """
        generated_tokens = self.model.generate(inputs, self.args.generation_max_length)
        
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < self.args.generation_max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, self.args.generation_max_length)

        loss = torch.tensor([0]).cuda() # we will never train llama through this function.

        labels = inputs["labels"]
        if labels.shape[-1] < self.args.generation_max_length:
            labels = self._pad_tensors_to_max_len(labels, self.args.generation_max_length)

        return (loss, generated_tokens, labels)


    def _post_process_function(
            self, examples: Dataset, predictions: np.ndarray, stage: str
    ) -> EvalPrediction:
        # assert isinstance(examples, Dataset)

        try:
            predictions = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        except:
            import pickle
            with open(f"{self.args.output_dir}/preds.pkl", "wb") as f:
                pickle.dump(predictions, f)
            pickle.dump(examples, open(f"{self.args.output_dir}/preds_examples.pkl", "wb"))
        # Save locally.
        if self.args.local_rank <= 0:
            with open(f"{self.args.output_dir}/predictions_{stage}.json", "w") as f:
                json.dump(
                    [dict(**{"prediction": predictions[idx]}, **examples[idx]) for idx in range(len(predictions))],
                    f,
                    indent=4,
                )

        # Save to wandb.
        if self.wandb_run_dir and self.args.local_rank <= 0:
            with open(f"{self.wandb_run_dir}/predictions_{stage}.json", "w") as f:
                json.dump(
                    [dict(**{"prediction": predictions[idx]}, **examples[idx]) for idx in range(len(predictions))],
                    f,
                    indent=4,
                )
        return EvalPrediction(predictions=predictions, items=[examples[idx] for idx in range(len(predictions))])

    def _compute_metrics(self, eval_prediction: EvalPrediction, section) -> dict:
        # try:
        return self.evaluator.evaluate(eval_prediction.predictions, eval_prediction.items, section)
        # except:
        #     import pdb; pdb.post_mortem()

