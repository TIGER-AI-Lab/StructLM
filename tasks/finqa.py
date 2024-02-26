'''
Author: ygjin11 1633504509@qq.com
Date: 2024-01-09 04:16:22
LastEditors: ygjin11 1633504509@qq.com
LastEditTime: 2024-02-14 09:51:20
FilePath: /uskg_eval/tasks/finqa.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The WikiTableQuestions dataset is a large-scale dataset for the task of question answering on semi-structured tables."""

import os

import datasets

import json

_HOMEPAGE = "https://finqasite.github.io/index.html"

_GIT_ARCHIVE_URL = (
    "https://github.com/czyssrs/FinQA/archive/refs/heads/main.zip"
)

class FinQA(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features(
            {
                # "filename": datasets.Value("string"),
                "id": datasets.Value("string"),
                "post_text": datasets.features.Sequence(datasets.Value("string")),
                "pre_text": datasets.features.Sequence(datasets.Value("string")),
                "question": datasets.Value("string"),
                "answer": datasets.Value("string"),
                "final_res": datasets.Value("string"),
                "steps": datasets.Value("string"),
                "program": datasets.Value("string"),
                "gold_evidence": datasets.features.Sequence(datasets.Value("string")),
                "table": datasets.features.Sequence(datasets.features.Sequence(datasets.Value("string"))),
            }
        )
        return datasets.DatasetInfo(
            features=features,
        )

    def _split_generators(self, dl_manager):
        extracted_path = dl_manager.download_and_extract(_GIT_ARCHIVE_URL)
        print(extracted_path)
        train_file = os.path.join(extracted_path, "FinQA-main", "dataset", "train.json")
        dev_file = os.path.join(extracted_path, "FinQA-main", "dataset", "dev.json")
        test_file = os.path.join(extracted_path, "FinQA-main", "dataset", "test.json")
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"main_filepath": train_file},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"main_filepath": dev_file},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"main_filepath": test_file},
            ),
        ]


    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, main_filepath):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(main_filepath, encoding="utf-8") as f:
            # skip the first line since it is the tsv header
            lines = json.load(f)
            for idx, example in enumerate(lines):
                yield idx, { 
                "id": example['id'], 
                "post_text": example['post_text'], 
                "pre_text": example['pre_text'], 
                "question": example['qa']['question'], ``
                "answer": example['qa']['answer'],
                "steps": str(example['qa']['steps']),
                "program": str(example['qa']['program']),
                'final_res': str(example['qa']['steps'][-1]['res']),
                "table": example['table'], 
                "gold_evidence": list(example['qa']['gold_inds'].values())
            }