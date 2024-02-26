import os
import torch
from torch.utils.data import Dataset
import json


class TokenizedDataset(Dataset):
    # TODO: A unified structure-representation.
    def __init__(self, args, training_args, tokenizer, seq2seq_dataset, ):
        self.args = args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.seq2seq_dataset = seq2seq_dataset

        self.conv_sep = " || "

    def __getitem__(self, index):
        raw_item = self.seq2seq_dataset[index]

        if raw_item["text_in"]:
            ###################
            # With text input #
            ###################
            if self.conv_sep in raw_item["text_in"]:
                ##################
                # Conversational #
                ##################
                # TODO (commented by Chen): the context part roughly follows the implementation of CoSQL by Tianbao.
                # text_in = "[utt n] || [utt n-1] | [utt n-2] | ..."
                index = raw_item["text_in"].index(self.conv_sep)
                if self.args.model.knowledge_usage == 'concatenate' or self.args.model.knowledge_usage is None:
                    # seq_in  = "[utt n] ; structured knowledge: struct_in ; context: [utt n-1] | [utt n-2] | ..."
                    seq_in = "{} ; structured knowledge: {} ; context: {}".format(raw_item["text_in"][:index],
                                                                                  raw_item["struct_in"],
                                                                                  raw_item["text_in"][index + len(self.conv_sep):])
                elif self.args.model.knowledge_usage == 'separate':
                    # seq_in  = "[utt n] ; context: [utt n-1] | [utt n-2] | ..."
                    seq_in = "{} ; context: {}".format(raw_item["text_in"][:index],
                                                       raw_item["text_in"][index + len(self.conv_sep):])
                else:
                    raise ValueError()
            else:
                ######################
                # Non-conversational #
                ######################
                if self.args.model.knowledge_usage == 'concatenate' or self.args.model.knowledge_usage is None:
                    # seq_in  = "text_in ; structured knowledge: struct_in"
                    seq_in = "{} ; structured knowledge: {}".format(raw_item["text_in"], raw_item["struct_in"])
                elif self.args.model.knowledge_usage == 'separate':
                    # seq_in  = "text_in"
                    seq_in = raw_item["text_in"]
                else:
                    raise ValueError()
        else:
            ######################
            # Without text input #
            ######################
            if self.args.model.knowledge_usage == 'concatenate':
                # seq_in  = "structured knowledge: struct_in"
                seq_in = "structured knowledge: {}".format(raw_item["struct_in"])
            elif self.args.model.knowledge_usage == 'separate':
                # seq_in  = ""
                seq_in = ""
            else:
                raise ValueError()

        # only run this block if we are evaluating
        if self.args.prompt:
            with open(self.args.prompt_spec.path, 'r') as f:
                prompt_spec = json.load(f)
            format_spec = prompt_spec[self.args.prompt_spec.dataset_name]
            if (type(format_spec['instruction']) == list):
                main_instruction = format_spec['instruction'][0]
            else:
                main_instruction = format_spec['instruction']
            if self.args.prompt.spec.dataset_name == "bird":
                seq_in = "{} ; {}".format(main_instruction, seq_in)
            elif self.args.prompt.spec.dataset_name == "cosql":
                seq_in = "{} ; {}".format(main_instruction, seq_in)
            elif self.args.prompt.spec.dataset_name == "sqa":
                seq_in = "{} ; {}".format(main_instruction, seq_in)
            elif self.args.prompt.spec.dataset_name == "infotabs":
                seq_in = "{} ; {}".format(main_instruction, seq_in)
            elif self.args.prompt.spec.dataset_name == "finqa":
                seq_in = "{} ; {}".format(main_instruction, seq_in)
            elif self.args.prompt.spec.dataset_name == "grailqa":
                seq_in = "{} ; {}".format(main_instruction, seq_in)
            else:
                if self.args.model.use_description and self.args.model.concatenate_description:
                    seq_in = "{} ; {}".format(raw_item["description"], seq_in)
        else:
            # Concatenate description.
            if self.args.model.use_description and self.args.model.concatenate_description:
                seq_in = "{} ; {}".format(raw_item["description"], seq_in)

        # if self.args.prompt_spec.dataset_name == "logicnlg":
        #     seq_in ="use the information in the table to infer a fact about the subjects in the table ; " + seq_in
        # elif self.args.prompt_spec.dataset_name == "finqa":
        #     seq_in += " ; let's think step by step:"
        # elif self.args.prompt_spec.dataset_name == "infotabs":
        #     pass

        tokenized_question_and_schemas = self.tokenizer(
            seq_in,
            padding="max_length",
            truncation=True,
            max_length=self.training_args.input_max_length,
            # We found that set it as large as possible can boost the performance significantly
            # , meanwhile, due to the t5 uses a relative position coding, we need to manually
            # assign the max input length into some large numbers, instead of using the "max_model_length"
            # ,which the default is 512, which will hurt the performance a lot.
        )
        tokenized_inferred = self.tokenizer(
            raw_item["seq_out"],
            padding="max_length",
            truncation=True,
            max_length=self.training_args.generation_max_length,
            # We set the max_length of "seq_out" during training is the same with the one in inference.
        )

        tokenized_inferred_input_ids = torch.LongTensor(tokenized_inferred.data["input_ids"])
        # Here -100 will let the model not to compute the loss of the padding tokens.
        tokenized_inferred_input_ids[tokenized_inferred_input_ids == self.tokenizer.pad_token_id] = -100

        item = {
            'input_ids': torch.LongTensor(tokenized_question_and_schemas.data["input_ids"]),
            'attention_mask': torch.LongTensor(tokenized_question_and_schemas.data["attention_mask"]),
            'labels': tokenized_inferred_input_ids,
        }
        # Add task name.
        if 'task_id' in raw_item:
            item['task_ids'] = raw_item['task_id']

        # Separate description tokenization.
        if self.args.model.use_description and self.args.model.map_description:
            tokenized_description = self.tokenizer(raw_item["description"],
                                                   padding="max_length",
                                                   truncation=True,
                                                   max_length=self.args.dataset.description_max_length,
                                                   )
            item['description_input_ids'] = torch.LongTensor(tokenized_description.data["input_ids"])
            item['description_attention_mask'] = torch.LongTensor(tokenized_description.data["attention_mask"])

        # Separate knowledge tokenization.
        if self.args.model.knowledge_usage == 'separate':
            tokenized_knowledge = self.tokenizer(raw_item["struct_in"],
                                                 padding="max_length",
                                                 truncation=True,
                                                 max_length=self.training_args.input_max_length,
                                                 )
            item['knowledge_input_ids'] = torch.LongTensor(tokenized_knowledge.data["input_ids"])
            item['knowledge_attention_mask'] = torch.LongTensor(tokenized_knowledge.data["attention_mask"])

        return item

    def __len__(self):
        return len(self.seq2seq_dataset)


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

class TokenizedLlamaDataset(Dataset):
    # TODO: A unified structure-representation.z
    def __init__(self, args, training_args, tokenizer, seq2seq_dataset, few_shot_path = ""):
        self.args = args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.seq2seq_dataset = seq2seq_dataset

        self.conv_sep = " || "

        if few_shot_path:
            with open(few_shot_path, "r") as f:
                self.few_shot_examples = json.load(f)
        # open the prompt specification file and get the format for the specified dataset
        self.prompt_spec_path = args.prompt_spec.path
        # load the json
        with open(self.prompt_spec_path, 'r') as f:
            prompt_spec = json.load(f)
        self.format_spec = prompt_spec[args.prompt_spec.dataset_name]

        # if this is an untrained model
        if few_shot_path:
            if "models--codellama--CodeLlama" in self.args.llama.model_path:
                self.instuning_format = (
                    "[INST] <<SYS>> Respond tersely, following the format of the example. Do not try to explain. Do not give extra context <</SYS>>\n{instruction}\nHere are some examples:\n<examples>\n\nInput: {input} [/INST] "
                )
            else:
                self.instuning_format = (
                    "Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n{instruction}\n\nHere are some examples:\n<examples>\n\n### Input:\n{input}\n\n### Response:\n"
                )
            self.instuning_format = self.instuning_format.replace("<examples>", self.few_shot_examples[args.prompt_spec.dataset_name]['ex'])
        else:
            if "models--codellama--CodeLlama" in self.args.llama.model_path:
                self.instuning_format = (
                    "[INST] <<SYS>> Respond tersely. Do not try to explain. Do not give extra context <</SYS>>\n{instruction}\n\n{input} [/INST] "
                )
            else:
                self.instuning_format = (
                    "Below is an instruction that describes a task, paired with an input that provides further context. "
                    "Write a response that appropriately completes the request.\n\n"
                    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
                )

        # self.instuning_format = (
        #     "[INST] {instruction}\n\n{input} [/INST] "
        # )
        # format the above string by insserting instruction
        if (type(self.format_spec['instruction']) == list):
            main_instruction = self.format_spec['instruction'][0]
        else:
            main_instruction = self.format_spec['instruction']
        input_format_str = self.format_spec['input_format'].replace('<struct_in>', '{struct_in}').replace('<text_in>', '{text_in}')
        self.instuning_format = self.instuning_format.format(instruction=main_instruction, input=input_format_str)

        # at this point, we have a format string that we just need to format the struct and the text input into.

    def __getitem__(self, index):
        raw_item = self.seq2seq_dataset[index]

        if raw_item["text_in"]:
            pre_struct = self.instuning_format.format(struct_in=raw_item['struct_in'], text_in=raw_item["text_in"])
        else:
            pre_struct = self.instuning_format.format(struct_in=raw_item['struct_in'])

        # if self.args.prompt_spec.dataset_name == "finqa":
        #     pre_struct += "Let's think step by step:"
        # get the char index of when the struct starts and ends
        struct_start = pre_struct.find(raw_item['struct_in'])
        struct_end = struct_start + len(raw_item['struct_in'])
        seq_in = self.tokenizer(pre_struct)

        start_struct_token = seq_in.char_to_token(struct_start)
        post_struct_token = seq_in.char_to_token(struct_end)
        diff = max(len(seq_in.input_ids) - self.training_args.input_max_length, 0)
        if (post_struct_token - start_struct_token <= diff and diff > 0):
            # this is not okay. 
            import pdb; pdb.set_trace()
        struct_end_token = post_struct_token - diff
        seq_in['input_ids'] = seq_in.input_ids[:struct_end_token] + seq_in.input_ids[post_struct_token:]
        seq_in['attention_mask'] = seq_in.attention_mask[:struct_end_token] + seq_in.attention_mask[post_struct_token:]
        
        tokenized_question_and_schemas = seq_in
        # ensure that we haven't truncated the input
        assert len(tokenized_question_and_schemas.input_ids) <= self.training_args.input_max_length

        tokenized_inferred = self.tokenizer(
            raw_item["seq_out"],
            #padding="max_length",
            truncation=True,
            max_length=self.training_args.generation_max_length,
            # We set the max_length of "seq_out" during training is the same with the `one in inference.
        )

        tokenized_inferred_input_ids = torch.LongTensor(tokenized_inferred.data["input_ids"])
        # Here -100 will let the model not to compute the loss of the padding tokens.
        tokenized_inferred_input_ids[tokenized_inferred_input_ids == self.tokenizer.pad_token_id] = -100

        item = {
            'input_ids': torch.LongTensor(tokenized_question_and_schemas.data["input_ids"]),
            'attention_mask': torch.LongTensor(tokenized_question_and_schemas.data["attention_mask"]),
            'labels': tokenized_inferred_input_ids,
        }
        # Add task name.
        if 'task_id' in raw_item:
            item['task_ids'] = raw_item['task_id']

        # Separate description tokenization.
        if self.args.model.use_description and self.args.model.map_description:
            tokenized_description = self.tokenizer(raw_item["description"],
                                                   #padding="max_length",
                                                   truncation=True,
                                                   max_length=self.args.dataset.description_max_length,
                                                   )
            item['description_input_ids'] = torch.LongTensor(tokenized_description.data["input_ids"])
            item['description_attention_mask'] = torch.LongTensor(tokenized_description.data["attention_mask"])

        # Separate knowledge tokenization.
        if self.args.model.knowledge_usage == 'separate':
            tokenized_knowledge = self.tokenizer(raw_item["struct_in"],
                                                 #padding="max_length",
                                                 truncation=True,
                                                 max_length=self.training_args.input_max_length,
                                                 )
            item['knowledge_input_ids'] = torch.LongTensor(tokenized_knowledge.data["input_ids"])
            item['knowledge_attention_mask'] = torch.LongTensor(tokenized_knowledge.data["attention_mask"])

        return item

    def __len__(self):
        return len(self.seq2seq_dataset)



class TokenizedT5LlamaDataset(Dataset):
    # TODO: A unified structure-representation.
    def __init__(self, args, training_args, tokenizer, seq2seq_dataset, few_shot_examples = []):
        self.args = args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.seq2seq_dataset = seq2seq_dataset

        self.conv_sep = " || "

        # open the prompt specification file and get the format for the specified dataset
        self.prompt_spec_path = args.prompt_spec.path
        # load the json
        with open(self.prompt_spec_path, 'r') as f:
            self.prompt_spec = json.load(f)


        dataset_names = ['bird', 'logicnlg', 'tabmwp', 'finqa', 'infotabs', 'totto', 'webqsp', 'sqa', 'sql2text', 'spider', 'cosql', 'kvret', 'hybridqa', 'sparc', 'grailqa', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop']
        datasets = ['bird', 'logicnlg', 'tabmwp', 'finqa', 'infotabs', 'totto', 'webqsp', 'sqa', 'sql2text', 'spider_with_cell', 'cosql_with_cell', 'kvret', 'hybridqa', 'sparc_with_cell', 'grailqa', 'compwebq', 'tab_fact', 'wikitq', 'wikisql', 'fetaqa', 'feverous', 'multiwoz', 'dart', 'logic2text', 'mtop']
        self.name_map = {datasets[i] : dataset_names[i] for i in range(len(datasets))}


        # if this is an untrained model
        if few_shot_examples:
            self.instuning_format = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request. "
                "### Instruction: {instruction} Here are some examples: {examples} ### Input: {input}"
            )
        else:
            self.instuning_format = (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request. "
                "### Instruction: {instruction} ### Input: {input}"
            )

        # self.instuning_format = (
        #     "[INST] {instruction}\n\n{input} [/INST] "
        # )
        # format the above string by insserting instruction

        # at this point, we have a format string that we just need to format the struct and the text input into.

    def __getitem__(self, index):
        
        raw_item = self.seq2seq_dataset[index]
        meta_tuning_name = os.path.basename(raw_item['arg_path']).split('.')[0]
        format_spec = self.prompt_spec[self.name_map[meta_tuning_name]]

        if (type(format_spec['instruction']) == list):
            main_instruction = format_spec['instruction'][0]
        else:
            main_instruction = format_spec['instruction']
        input_format_str = format_spec['input_format'].replace('<struct_in>', '{struct_in}').replace('<text_in>', '{text_in}')
        instuning_format = self.instuning_format.format(instruction=main_instruction, input=input_format_str)

        if raw_item["text_in"]:
            pre_struct = instuning_format.format(struct_in=raw_item['struct_in'], text_in=raw_item["text_in"])
        else:
            pre_struct = instuning_format.format(struct_in=raw_item['struct_in'])

        tokenized_question_and_schemas = self.tokenizer(
            pre_struct,
            padding="max_length",
            truncation=True,
            max_length=self.training_args.input_max_length,
            # We found that set it as large as possible can boost the performance significantly
            # , meanwhile, due to the t5 uses a relative position coding, we need to manually
            # assign the max input length into some large numbers, instead of using the "max_model_length"
            # ,which the default is 512, which will hurt the performance a lot.
        )
        tokenized_inferred = self.tokenizer(
            raw_item["seq_out"],
            padding="max_length",
            truncation=True,
            max_length=self.training_args.generation_max_length,
            # We set the max_length of "seq_out" during training is the same with the one in inference.
        )

        tokenized_inferred_input_ids = torch.LongTensor(tokenized_inferred.data["input_ids"])
        # Here -100 will let the model not to compute the loss of the padding tokens.
        tokenized_inferred_input_ids[tokenized_inferred_input_ids == self.tokenizer.pad_token_id] = -100

        item = {
            'input_ids': torch.LongTensor(tokenized_question_and_schemas.data["input_ids"]),
            'attention_mask': torch.LongTensor(tokenized_question_and_schemas.data["attention_mask"]),
            'labels': tokenized_inferred_input_ids,
        }

        # Add task name.
        if 'task_id' in raw_item:
            item['task_ids'] = raw_item['task_id']

        # Separate description tokenization.
        if self.args.model.use_description and self.args.model.map_description:
            tokenized_description = self.tokenizer(raw_item["description"],
                                                   #padding="max_length",
                                                   truncation=True,
                                                   max_length=self.args.dataset.description_max_length,
                                                   )
            item['description_input_ids'] = torch.LongTensor(tokenized_description.data["input_ids"])
            item['description_attention_mask'] = torch.LongTensor(tokenized_description.data["attention_mask"])

        # Separate knowledge tokenization.
        if self.args.model.knowledge_usage == 'separate':
            tokenized_knowledge = self.tokenizer(raw_item["struct_in"],
                                                 #padding="max_length",
                                                 truncation=True,
                                                 max_length=self.training_args.input_max_length,
                                                 )
            item['knowledge_input_ids'] = torch.LongTensor(tokenized_knowledge.data["input_ids"])
            item['knowledge_attention_mask'] = torch.LongTensor(tokenized_knowledge.data["attention_mask"])

        return item

    def __len__(self):
        return len(self.seq2seq_dataset)

