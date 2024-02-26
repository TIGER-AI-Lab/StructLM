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
import pandas as pd
import json

_HOMEPAGE = "https://finqasite.github.io/index.html"

_GIT_ARCHIVE_URL = (
    "https://github.com/infotabs/infotabs/archive/refs/heads/master.zip"
)

class Infotabs(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("1.0.0")

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.table_cache= dict()

    def _info(self):
        features = datasets.Features(
            {
                # "filename": datasets.Value("string"),
                "table_id": datasets.Value("string"),
                "hypothesis": datasets.Value("string"),
                "table": datasets.Value("string"),
                "natural_label": datasets.Value("string"),
                "label": datasets.Value("string"),
            }
        )
        return datasets.DatasetInfo(
            features=features,
        )

    def _split_generators(self, dl_manager):
        extracted_path = dl_manager.download_and_extract(_GIT_ARCHIVE_URL)
        print(extracted_path)
        train_file = os.path.join(extracted_path, "infotabs-master/data/maindata/infotabs_train.tsv")
        dev_file = os.path.join(extracted_path, "infotabs-master/data/maindata/infotabs_dev.tsv")
        test_files = [os.path.join(extracted_path, "infotabs-master/data/maindata/infotabs_test_alpha1.tsv"),
                        os.path.join(extracted_path, "infotabs-master/data/maindata/infotabs_test_alpha2.tsv"),
                        os.path.join(extracted_path, "infotabs-master/data/maindata/infotabs_test_alpha3.tsv")]
        table_path = os.path.join(extracted_path, "infotabs-master/data/tables/json")
        
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"main_filepaths": [train_file], "table_path": table_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"main_filepaths": [dev_file], "table_path": table_path},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"main_filepaths": test_files, "table_path": table_path},
            ),
        ]


    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, main_filepaths, table_path):
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.
        # Make an empty dataframe to begin with
        dfs = []
        for main_filepath in main_filepaths:
            # read the tsv files and combine them into one huge pd dataframe
            dfs.append(pd.read_csv(main_filepath, sep="\t", encoding="utf-8"))
        df = pd.concat(dfs, ignore_index=True)

        for idx, row in df.iterrows():
            # get the table id
            table_id = row['table_id']
            # check if we have already cached the table
            if table_id not in self.table_cache:
                # if not, load it from disk
                with open(os.path.join(table_path, table_id + ".json"), "r") as f:
                    self.table_cache[table_id] = json.load(f)
            # get the table
            table = self.table_cache[table_id]
            if row['label'].lower() == "c":
                label = "wrong"
            elif row['label'].lower() == "e":
                label = "reasonable"
            else:
                label = "not necessarily either"
            yield idx, { 
                "table_id": table_id,
                "hypothesis": row['hypothesis'],
                "table": str(table),
                "natural_label": label,
                "label": row['label']
            }