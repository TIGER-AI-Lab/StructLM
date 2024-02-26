# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors and the current dataset script contributor.
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
"""Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs"""


import json
import sqlite3
import sys
import traceback

import datasets


logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@misc{li2023llm,
  title={Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs},
  author={Jinyang Li and Binyuan Hui and Ge Qu and Binhua Li and Jiaxi Yang and Bowen Li and Bailin Wang and Bowen Qin and Rongyu Cao and Ruiying Geng and Nan Huo and Chenhao Ma and Kevin C. C. Chang and Fei Huang and Reynold Cheng and Yongbin Li},
  year={2023},
  eprint={2305.03111},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
"""

_DESCRIPTION = """\
BIRD (BIg Bench for LaRge-scale Database Grounded Text-to-SQL Evaluation) represents a pioneering, cross-domain dataset that examines the impact of extensive database contents on text-to-SQL parsing. BIRD contains over 12,751 unique question-SQL pairs, 95 big databases with a total size of 33.4 GB. It also covers more than 37 professional domains, such as blockchain, hockey, healthcare and education, etc.
"""

_HOMEPAGE = "https://bird-bench.github.io/"

_LICENSE = "CC BY-NC 4.0"  # non commercial


_URL = "https://drive.google.com/uc?export=download&id=1bXF3zsKqso6zfweZ2EfyNo9z9WpX6M74"


            #     "query" : datasets.Value("string"),
            #     "question": datasets.Value("string"),
            #     "difficulty": datasets.Value("string"),
            #     "db_id": datasets.Value("string"),
            #     "db_path": datasets.Value("string"),
            #     "db_table_names": datasets.features.Sequence(datasets.Value("string")),
            #     "db_column_names": datasets.features.Sequence(
            #         {
            #             "table_id": datasets.Value("int32"),
            #             "column_name": datasets.Value("string"),
            #         }
            #     ),
            #     "db_column_types": datasets.features.Sequence(datasets.Value("string")),
            #     "db_primary_keys": datasets.features.Sequence({"column_id": datasets.Value("int32")}),
            #     "db_foreign_keys": datasets.features.Sequence(
            #         {
            #             "column_id": datasets.Value("int32"),
            #             "other_column_id": datasets.Value("int32"),
            #         }
            #     ),
            # }

class TabMWP(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="tabmwp",
            version=VERSION,
            description="",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()

    def _info(self):

        features = datasets.Features(
            {
                'question': datasets.Value("string"),
                # choices is either a sequence of strings or none
                'choices': datasets.Value("string"),
                'answer': datasets.Value("string"),
                'unit': datasets.Value("string"),
                'table_title': datasets.Value("string"),
                'table': datasets.Value("string"),
                # table for pd is a dictionary of lists. we will just store the string in here and eval it later as a hack
                'table_for_pd': datasets.Value("string"),
                'row_num': datasets.Value("int32"),
                'column_num': datasets.Value("int32"),
                'solution': datasets.Value("string"),
                'ques_type': datasets.Value("string"),
                'ans_type': datasets.Value("string"),
                'grade': datasets.Value("int32"),
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        # downloaded_filepath_train = dl_manager.download_and_extract(_TRAINURL)
        # TODO: we only load the dev data for now, we are holding out this dataset
        downloaded_filepath = dl_manager.download_and_extract(_URL)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/tabmwp/problems_train.json",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepath": downloaded_filepath + "/tabmwp/problems_dev.json",
                },
            ),
        ]

    def _generate_examples(self, data_filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", data_filepath)
        with open(data_filepath, encoding="utf-8") as f:
            data = json.load(f)
            for k, v in data.items():
                # tabmwp data is 1 indexed for some reason
                yield int(k)-1, {
                    'question': v['question'],
                    'choices': v['choices'],
                    'answer': v['answer'],
                    'unit': v['unit'],
                    'table_title': v['table_title'],
                    'table': v['table'],
                    'table_for_pd': str(v['table_for_pd']), # Hack
                    'row_num': v['row_num'],
                    'column_num': v['column_num'],
                    'solution': v['solution'],
                    'ques_type': v['ques_type'],
                    'ans_type': v['ans_type'],
                    'grade': v['grade'],
                }