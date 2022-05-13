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

"""Tests for SGD text data generation."""

import filecmp
import os

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized
from state_tracking.d3st import create_sgd_schemaless_data
import tensorflow as tf

FLAGS = flags.FLAGS

TEST_DIR = 'zero_shot_task_oriented_dialog/testdata'


class CreateSgdTxtTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      {
          'testcase_name': 'create_sgd_schemaless_dst',
          'level': 'dst',
          'data_format': 'full_desc',
      }, {
          'testcase_name': 'create_sgd_schemaless_dst_intent',
          'level': 'dst_intent',
          'data_format': 'full_desc',
      })
  def testGenerateDataFullDesc(self, level, data_format):
    temp_output = os.path.join(self.create_tempdir(), 'output')
    ref_output = os.path.join(FLAGS.test_srcdir, TEST_DIR,
                              f'sgd_text_v2_full_desc_{level}')
    with flagsaver.flagsaver(
        level=level,
        delimiter='=',
        data_format=data_format,
        sgd_file=os.path.join(FLAGS.test_srcdir, TEST_DIR, 'sgd_train.json'),
        schema_file=os.path.join(FLAGS.test_srcdir, TEST_DIR,
                                 'sgd_train_schema.json'),
        output_file=temp_output,
        randomize_items=False):
      slots, item_desc = create_sgd_schemaless_data.load_schema()
      create_sgd_schemaless_data.generate_data(slots, item_desc)
      self.assertTrue(filecmp.cmp(temp_output, ref_output))

  @parameterized.named_parameters(
      {
          'testcase_name': 'create_sgd_schemaless_dst',
          'level': 'dst',
          'data_format': 'item_name',
      }, {
          'testcase_name': 'create_sgd_schemaless_dst_intent',
          'level': 'dst_intent',
          'data_format': 'item_name',
      })
  def testGenerateDataItemName(self, level, data_format):
    temp_output = os.path.join(self.create_tempdir(), 'output')
    ref_output = os.path.join(FLAGS.test_srcdir, TEST_DIR,
                              f'sgd_text_v2_item_name_{level}')
    with flagsaver.flagsaver(
        level=level,
        delimiter='=',
        data_format=data_format,
        sgd_file=os.path.join(FLAGS.test_srcdir, TEST_DIR, 'sgd_train.json'),
        schema_file=os.path.join(FLAGS.test_srcdir, TEST_DIR,
                                 'sgd_train_schema.json'),
        output_file=temp_output,
        randomize_items=False):
      slots, item_desc = create_sgd_schemaless_data.load_schema()
      create_sgd_schemaless_data.generate_data(slots, item_desc)
      self.assertTrue(filecmp.cmp(temp_output, ref_output))

  @parameterized.named_parameters(
      {
          'testcase_name': 'create_sgd_schemaless_dst',
          'level': 'dst',
          'data_format': 'full_desc',
      }, {
          'testcase_name': 'create_sgd_schemaless_dst_intent',
          'level': 'dst_intent',
          'data_format': 'full_desc',
      })
  def testMultipleChoice(self, level, data_format):
    temp_output = os.path.join(self.create_tempdir(), 'output')
    ref_output = os.path.join(FLAGS.test_srcdir, TEST_DIR,
                              f'sgd_text_v2_multiple_choice_{level}')
    with flagsaver.flagsaver(
        level=level,
        delimiter='=',
        data_format=data_format,
        sgd_file=os.path.join(FLAGS.test_srcdir, TEST_DIR,
                              'sgd_train_categorical.json'),
        schema_file=os.path.join(FLAGS.test_srcdir, TEST_DIR,
                                 'sgd_train_schema.json'),
        output_file=temp_output,
        randomize_items=False,
        multiple_choice='1a'):
      slots, item_desc = create_sgd_schemaless_data.load_schema()
      create_sgd_schemaless_data.generate_data(slots, item_desc)
      self.assertTrue(filecmp.cmp(temp_output, ref_output))

  @parameterized.named_parameters(
      {
          'testcase_name': 'create_uniform_domain_10percent',
          'level': 'dst_intent',
          'data_format': 'full_desc',
          'data_percent': 0.1,
          'uniform_domain_distribution': True,
      }, {
          'testcase_name': 'create_random_domain_50percent',
          'level': 'dst_intent',
          'data_format': 'full_desc',
          'data_percent': 0.5,
          'uniform_domain_distribution': False,
      })
  def testGenerateDataSample(self, level, data_format, data_percent,
                             uniform_domain_distribution):
    temp_output = os.path.join(self.create_tempdir(), 'output')
    ref_output = os.path.join(
        FLAGS.test_srcdir, TEST_DIR,
        f'sgd_text_v2_uniform_{uniform_domain_distribution}_{data_percent}')
    with flagsaver.flagsaver(
        level=level,
        delimiter='=',
        data_format=data_format,
        sgd_file=os.path.join(FLAGS.test_srcdir, TEST_DIR, 'sgd_train.json'),
        schema_file=os.path.join(FLAGS.test_srcdir, TEST_DIR,
                                 'sgd_train_schema.json'),
        output_file=temp_output,
        randomize_items=False,
        data_percent=data_percent,
        uniform_domain_distribution=uniform_domain_distribution):
      slots, item_desc = create_sgd_schemaless_data.load_schema()
      create_sgd_schemaless_data.generate_data(slots, item_desc)
      self.assertTrue(filecmp.cmp(temp_output, ref_output))


if __name__ == '__main__':
  tf.test.main()
