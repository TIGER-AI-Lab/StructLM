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
import pandas as pd

_HOMEPAGE = ""

_TRAIN_URL = 'https://raw.githubusercontent.com/msra-nlc/Table2Text/master/MSRA_NLC.Table2Text.train'
_DEV_URL = 'https://raw.githubusercontent.com/msra-nlc/Table2Text/master/MSRA_NLC.Table2Text.dev'
_TEST_URL = 'https://raw.githubusercontent.com/msra-nlc/Table2Text/master/MSRA_NLC.Table2Text.test'

class WikiTableText(datasets.GeneratorBasedBuilder):

    # adapted from https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/structured/wiki_table_text/wiki_table_text.py
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        features = datasets.Features(
            {
                # "filename": datasets.Value("string"),
                "table": datasets.Sequence({
                    'column_header': datasets.Value("string"),
                    'row_number': datasets.Value("int16"),
                    'content': datasets.Value("string"),
                }),
                "target_text": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            features=features,
        )

    def _split_generators(self, dl_manager):
        extracted_path = dl_manager.download_and_extract(
            {'train_path': _TRAIN_URL, 'dev_path': _DEV_URL, 'test_path': _TEST_URL}
        )

        print(extracted_path)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"path": extracted_path['train_path']},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"path": extracted_path['dev_path']},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"path": extracted_path['test_path']},
            ),
        ]


    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, path):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
            for i, example_line in enumerate(lines):
                _, headers, values, text = example_line.split('\t')
                headers = [h.replace('_$$_', ' ') for h in headers.split('_||_')]
                values = [v.replace('_$$_', ' ') for v in values.split('_||_')]
                # The tables only have one row and we specify it because the dataset
                # follows a standarized table format.
                table = []
                for header_i, value_i in zip(headers, values):
                    table.append(
                        {'column_header': header_i, 'row_number': 1, 'content': value_i}
                    )
                text = text.replace('_$$_', ' ').replace(' .', '')
                yield i, {'table': table, 'target_text': text}
