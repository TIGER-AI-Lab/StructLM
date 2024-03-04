import os
import torch
from torch.utils.data import Dataset
import json


class TokenizedTestDataset(Dataset):
    # lightweight dataset class for testing
    def __init__(self, args, training_args, tokenizer, seq2seq_dataset):
        self.args = args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.seq2seq_dataset = seq2seq_dataset

    def __getitem__(self, index):
        raw_item = self.seq2seq_dataset[index]
        tokenized_question_and_schemas = self.tokenizer(raw_item["formatted_input"], padding = "max_length", max_length = self.training_args.input_max_length)
        assert len(tokenized_question_and_schemas.input_ids) == self.training_args.input_max_length, f"input length is {len(tokenized_question_and_schemas.input_ids)}, formatted input is {raw_item['formatted_input']}"

        tokenized_inferred = self.tokenizer(raw_item["seq_out"], padding = "max_length", max_length = self.training_args.generation_max_length, )
        assert len(tokenized_inferred.input_ids) == self.training_args.generation_max_length

        tokenized_inferred_input_ids = torch.LongTensor(tokenized_inferred.data["input_ids"])
        # Here -100 will let the model not to compute the loss of the padding tokens.
        tokenized_inferred_input_ids[tokenized_inferred_input_ids == self.tokenizer.pad_token_id] = -100

        item = {
            'input_ids': torch.LongTensor(tokenized_question_and_schemas.data["input_ids"]),
            'attention_mask': torch.LongTensor(tokenized_question_and_schemas.data["attention_mask"]),
            'labels': tokenized_inferred_input_ids,
        }
        return item
    
    def __len__(self):
        return len(self.seq2seq_dataset)