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

"""Tests for create_multiwoz_sdt_data."""

import json
import os

from absl import flags
from absl.testing import parameterized
from state_tracking.show_dont_tell import create_multiwoz_sdt_data
from state_tracking.utils import multiwoz_utils
import tensorflow as tf

FLAGS = flags.FLAGS

TEST_DIR = 'task-oriented-dialog/testdata'


class CreateMultiwozShowDontTellDataTest(tf.test.TestCase,
                                         parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._temp_dir = self.create_tempdir()
    self._testdata_dir = os.path.join(FLAGS.test_srcdir, TEST_DIR)
    self._schema_file = os.path.join(self._temp_dir, 'schema.json')

    # Setup data - general
    tf.io.gfile.copy(
        os.path.join(self._testdata_dir, 'multiwoz_slot_descriptions.json'),
        os.path.join(self._temp_dir, 'slot_descriptions.json'))
    tf.io.gfile.copy(
        os.path.join(self._testdata_dir, 'multiwoz_schema_schemaless.json'),
        self._schema_file)

    # Setup data - for 2.1 TRADE preprocessed data
    for target_file in [
        'train_dials.json', 'dev_dials.json', 'test_dials.json'
    ]:
      tf.io.gfile.copy(
          os.path.join(self._testdata_dir, 'multiwoz_data_trade.json'),
          os.path.join(self._temp_dir, target_file))

    # Setup data - for non-TRADE 2.2-2.4 data
    tf.io.gfile.copy(
        os.path.join(self._testdata_dir, 'multiwoz_data.json'),
        os.path.join(self._temp_dir, 'data.json'))

    # Touch empty files for (val|test)ListFile.json
    def _touch_empty_file(path):
      with tf.io.gfile.GFile(path, 'w') as f:
        f.write('')

    _touch_empty_file(os.path.join(self._temp_dir, 'valListFile.json'))
    _touch_empty_file(os.path.join(self._temp_dir, 'testListFile.json'))

  @parameterized.named_parameters(
      dict(
          testcase_name='all_domains',
          use_active_domains_only=False,
          blocked_domains=set(),
          mcq_cat_vals=False,
          multiwoz_version='2.4',
          ref_output_filename='mw24_sdt_all_domains.json',
          expected_len=7),
      dict(
          testcase_name='active_domains',
          use_active_domains_only=True,
          blocked_domains=set(),
          mcq_cat_vals=False,
          multiwoz_version='2.4',
          ref_output_filename='mw24_sdt_active_domains.json',
          expected_len=7),
      dict(
          testcase_name='all_domains_cat_val_mcq',
          use_active_domains_only=False,
          blocked_domains=set(),
          mcq_cat_vals=True,
          multiwoz_version='2.4',
          ref_output_filename='mw24_sdt_all_domains_cat_val_mcq.json',
          expected_len=7),
      dict(
          testcase_name='active_domains_block_restaurant',
          use_active_domains_only=True,
          blocked_domains={'hotel'},
          mcq_cat_vals=False,
          multiwoz_version='2.4',
          ref_output_filename='mw24_sdt_active_domains_block_hotel.json',
          expected_len=4),
      dict(
          testcase_name='all_domains_21',
          use_active_domains_only=False,
          blocked_domains=set(),
          mcq_cat_vals=False,
          multiwoz_version='2.1',
          ref_output_filename='mw21_trade_sdt_all_domains.json',
          expected_len=7))
  def test_generate_data(self, use_active_domains_only, blocked_domains,
                         mcq_cat_vals, multiwoz_version, ref_output_filename,
                         expected_len):
    # The following is needed so that assertEqual shows the full diff.
    self.maxDiff = None  # pylint:disable=invalid-name

    ref_output = os.path.join(self._testdata_dir, 'show_dont_tell',
                              ref_output_filename)

    is_trade = True if multiwoz_version == '2.1' else False
    multiwoz_data = multiwoz_utils.load_data(self._temp_dir, multiwoz_version,
                                             is_trade)
    examples = create_multiwoz_sdt_data.create_sdt_examples(
        multiwoz_data.train_json,
        create_multiwoz_sdt_data.Options(
            multiwoz_version=multiwoz_version,
            is_trade=is_trade,
            prompt_format='separated',
            prompt_indices=['0'],
            context_format='dialogue',
            target_format='all',
            randomize_slots=False,
            use_active_domains_only=use_active_domains_only,
            blocked_domains=blocked_domains,
            mcq_cat_vals=mcq_cat_vals,
            randomize_cat_vals=False,
            lowercase=True))

    self.assertLen(examples, expected_len)

    # Reference json contains 2nd and last examples we expect to generate
    with tf.io.gfile.GFile(ref_output) as ref_f:
      ref_json = json.load(ref_f)

    # Compare the 2nd example generated
    self.assertEqual(examples[1].src, ref_json[0]['src'])
    self.assertEqual(examples[1].tgt, ref_json[0]['tgt'])
    self.assertEqual(examples[1].dialog_id, ref_json[0]['dialog_id'])
    self.assertEqual(examples[1].turn, ref_json[0]['turn'])

    # Compare the last example generated
    self.assertEqual(examples[-1].src, ref_json[1]['src'])
    self.assertEqual(examples[-1].tgt, ref_json[1]['tgt'])
    self.assertEqual(examples[-1].dialog_id, ref_json[1]['dialog_id'])
    self.assertEqual(examples[-1].turn, ref_json[1]['turn'])


if __name__ == '__main__':
  tf.test.main()
