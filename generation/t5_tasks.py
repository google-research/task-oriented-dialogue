# Copyright 2021 Google Research.
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

"""Add Tasks to registry."""

import functools

import t5.data
from t5.data import preprocessors
from t5.evaluation import metrics


# This is the value of the flag `output_dir` used while executing
# `prepare_dataset.py`.
TSV_DATA_DIR = "/tmp/test_nlg"

for input_version in ["naive", "schema_guided", "t2g2"]:
  for kshot in ["5_shot", "10_shot", "20_shot", "40_shot", "80_shot", "all"]:
    t5.data.TaskRegistry.add(
        f"{input_version}_{kshot}",
        t5.data.TextLineTask,
        text_preprocessor=functools.partial(
            preprocessors.preprocess_tsv, num_fields=5),
        split_to_filepattern={
            "train": f"{TSV_DATA_DIR}/train/{input_version}_{kshot}.tsv",
            "dev": f"{TSV_DATA_DIR}/dev/{input_version}_all.tsv",
            "test": f"{TSV_DATA_DIR}/test/{input_version}_all.tsv",
        },
        metric_fns=[metrics.bleu, metrics.sequence_accuracy])
