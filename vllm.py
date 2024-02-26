#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
import transformers 
import requests
import torch
from vllm import LLM, SamplingParams


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not self.args.model.tp_size:
            self.args.model.tp_size = 1

        # padding side right is a minor hack to make the tokenizer compatible with the rest of the HF Trainer API as we have written it.
        tokenizer = transformers.AutoTokenizer.from_pretrained(args.model.path, use_fast=True, add_eos_token=False, padding_side = "right")
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        print("tensor parallel size, ", self.args.model.tp_size)
        self.model = LLM(model=args.model.path, tensor_parallel_size=self.args.model.tp_size) 
        self.tokenizer = tokenizer

    def forward(self, input_ids, attention_mask, labels):
        return {'loss': 0} # we don't need to compute loss for this model

    def generate(self, inputs, generation_max_length, **kwargs):


        input_ids = inputs["input_ids"].tolist()
        # unpad the input_ids
        for i in range(len(input_ids)):
            if self.tokenizer.pad_token_id in input_ids[i]:
                end_token_idx = input_ids[i].index(self.tokenizer.pad_token_id)
            else:
                end_token_idx = len(input_ids[i])
            input_ids[i] = input_ids[i][:end_token_idx]
        # add the eos token back to the end

        sampling_params = SamplingParams(temperature=0, max_tokens=generation_max_length, skip_special_tokens=False)
        generations = self.model.generate(prompt_token_ids = input_ids, sampling_params=sampling_params, use_tqdm=False)

        token_ids = [each.outputs[0].token_ids for each in generations]
        # convert ragged list of token_ids to batched tensor, This is technically a hack for compatibility with the rest of the HF Trainer API
        
        generated_tokens = torch.nn.utils.rnn.pad_sequence([torch.tensor(each) for each in token_ids], batch_first=True, padding_value=self.tokenizer.pad_token_id)

        return generated_tokens