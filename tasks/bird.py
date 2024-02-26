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


_TRAINURL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"
_DEVURL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip"


def convert_fk_index(data):
    fk_holder = []
    for fk in data["foreign_keys"]:
        tn, col, ref_tn, ref_col = fk[0][0], fk[0][1], fk[1][0], fk[1][1]
        ref_cid, cid = None, None
        try:
            # we gotta make this case insensitive
            lookup = [each.lower() for each in data['table_names_original']]
            tid = lookup.index(tn.lower())
            ref_tid = lookup.index(ref_tn.lower())

            for i, (tab_id, col_org) in enumerate(data["column_names_original"]):
                if tab_id == ref_tid and ref_col == col_org:
                    ref_cid = i
                elif tid == tab_id and col == col_org:
                    cid = i
            if ref_cid and cid:
                fk_holder.append([cid, ref_cid])
        except:
            traceback.print_exc()
            print("table_names_original: ", data["table_names_original"])
            print("finding tab name: ", tn, ref_tn)
            sys.exit()
    return fk_holder


def dump_db_json_schema(db, f):
    """read table and column info"""

    conn = sqlite3.connect(db)
    conn.execute("pragma foreign_keys=ON")
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")

    data = {
        "db_id": f,
        "table_names_original": [],
        "table_names": [],
        "column_names_original": [(-1, "*")],
        "column_names": [(-1, "*")],
        "column_types": ["text"],
        "primary_keys": [],
        "foreign_keys": [],
    }

    fk_holder = []
    for i, item in enumerate(cursor.fetchall()):
        table_name = item[0]
        data["table_names_original"].append(table_name)
        data["table_names"].append(table_name.lower().replace("_", " "))
        fks = conn.execute(
            "PRAGMA foreign_key_list('{}') ".format(table_name)
        ).fetchall()
        # print("db:{} table:{} fks:{}".format(f,table_name,fks))
        fk_holder.extend([[(table_name, fk[3]), (fk[2], fk[4])] for fk in fks])
        cur = conn.execute("PRAGMA table_info('{}') ".format(table_name))
        for j, col in enumerate(cur.fetchall()):
            data["column_names_original"].append((i, col[1]))
            data["column_names"].append((i, col[1].lower().replace("_", " ")))
            # varchar, '' -> text, int, numeric -> integer,
            col_type = col[2].lower()
            if (
                "char" in col_type
                or col_type == ""
                or "text" in col_type
                or "var" in col_type
            ):
                data["column_types"].append("text")
            elif (
                "int" in col_type
                or "numeric" in col_type
                or "decimal" in col_type
                or "number" in col_type
                or "id" in col_type
                or "real" in col_type
                or "double" in col_type
                or "float" in col_type
            ):
                data["column_types"].append("number")
            elif "date" in col_type or "time" in col_type or "year" in col_type:
                data["column_types"].append("time")
            elif "boolean" in col_type:
                data["column_types"].append("boolean")
            else:
                data["column_types"].append("others")

        if col[5] == 1:
                data["primary_keys"].append(len(data["column_names"]) - 1)

    data["foreign_keys"] = fk_holder
    data["foreign_keys"] = convert_fk_index(data)

    return data

class BIRD(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="bird",
            version=VERSION,
            description="A BIg Bench for Large-Scale Database Grounded Text-to-SQLs",
        ),
    ]

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        self.schema_cache = dict()

    def _info(self):
        features = datasets.Features(
            {
                "query" : datasets.Value("string"),
                "question": datasets.Value("string"),
                #"difficulty": datasets.Value("string"),
                "db_id": datasets.Value("string"),
                "db_path": datasets.Value("string"),
                "db_table_names": datasets.features.Sequence(datasets.Value("string")),
                "db_column_names": datasets.features.Sequence(
                    {
                        "table_id": datasets.Value("int32"),
                        "column_name": datasets.Value("string"),
                    }
                ),
                "db_column_types": datasets.features.Sequence(datasets.Value("string")),
                "db_primary_keys": datasets.features.Sequence({"column_id": datasets.Value("int32")}),
                "db_foreign_keys": datasets.features.Sequence(
                    {
                        "column_id": datasets.Value("int32"),
                        "other_column_id": datasets.Value("int32"),
                    }
                ),
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
        downloaded_filepath_train = dl_manager.download_and_extract(_TRAINURL)
        downloaded_database_train = dl_manager.extract(
            downloaded_filepath_train + "/train/train_databases.zip")
        downloaded_filepath_dev = dl_manager.download_and_extract(_DEVURL)
        downloaded_database_dev = dl_manager.extract(
            downloaded_filepath_dev + "/dev_20230613/dev_databases.zip")

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_filepath": downloaded_filepath_train + "/train/train.json",
                    "db_path": downloaded_database_train+"/train_databases",
                    "schema_path":downloaded_filepath_train+"/train/train_tables.json"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_filepath": downloaded_filepath_dev + "/dev_20230613/dev.json",
                    "db_path": downloaded_database_dev+"/dev_databases",
                    "schema_path":downloaded_filepath_dev+"/dev_20230613/dev_tables.json"
                },
            ),
        ]

    def _generate_examples(self, data_filepath, db_path, schema_path):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", data_filepath)
        with open(data_filepath, encoding="utf-8") as f:
            data = json.load(f)
            for i, sample in enumerate(data):
                db_id = sample['db_id']
                if db_id not in self.schema_cache:
                        self.schema_cache[db_id] = dump_db_json_schema(
                            db_path + "/" + db_id + "/" + db_id + ".sqlite", db_id
                        )
                schema = self.schema_cache[db_id]
                #schema = dump_db_json_schema(db_path + "/" + data[i]['db_id'] + "/" + data[i]['db_id'] + ".sqlite", data[i]['db_id'])
                #schema = ts_map[data[i]['db_id']]
                yield i, {
                    "query": sample["SQL"],
                    "question": sample["question"],
                    #"difficulty": sample["difficulty"],
                    "db_id": sample["db_id"],
                    "db_path": db_path + "/" + data[i]['db_id'] + "/" + data[i]['db_id'] + ".sqlite",
                    "db_table_names": schema["table_names_original"],
                    "db_column_names": [
                        {"table_id": table_id, "column_name": column_name}
                        for table_id, column_name in schema["column_names_original"]
                    ],
                    "db_column_types": schema["column_types"],
                    "db_primary_keys": [{"column_id": column_id} for column_id in schema["primary_keys"]],
                    "db_foreign_keys": [
                        {"column_id": column_id, "other_column_id": other_column_id}
                        for column_id, other_column_id in schema["foreign_keys"]
                    ],
                }