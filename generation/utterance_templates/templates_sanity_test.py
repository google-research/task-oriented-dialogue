# Copyright 2020 Google Research.
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

# Lint as: python3
"""Validates all utterance templates added to this directory.

Verifies that all rows have 2 columns separated by tabs and the number of slot
placeholders match in the template key and the template string.
"""

import os

from absl.testing import absltest
from absl.testing import parameterized


def get_template_paths():
  template_dir = os.path.dirname(os.path.abspath(__file__))
  template_paths = []
  for root, _, files in os.walk(template_dir):
    template_paths.extend(
        os.path.join(root, f) for f in files if f.endswith(".tsv"))
  return template_paths


@parameterized.parameters(get_template_paths())
class TemplateSanityTest(parameterized.TestCase):

  def test_valid_rows(self, template_path):
    with open(template_path) as f:
      for line in f:
        vals = line.strip().split("\t")
        self.assertLen(vals, 2)
        self.assertEqual(vals[0].count("@"), vals[1].count("@"))


if __name__ == "__main__":
  absltest.main()
