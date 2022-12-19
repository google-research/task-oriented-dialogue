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

"""Tests for create_sgd_sdt_data."""

import os

from absl import flags
from absl.testing import flagsaver
from absl.testing import parameterized
from state_tracking.show_dont_tell import create_sgd_sdt_data
import tensorflow as tf

FLAGS = flags.FLAGS

TEST_DIR = 'task-oriented-dialog/testdata'


class CreateSchemaGuidedDialogueShowDontTellDataTest(tf.test.TestCase,
                                                     parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # The following is needed so that assertEqual shows the full diff.
    self.maxDiff = None  # pylint:disable=invalid-name
    self._tempdir = self.create_tempdir()
    self._testdata_dir = os.path.join(FLAGS.test_srcdir, TEST_DIR)

  @parameterized.named_parameters(
      ('all_slots_slot_names', 'all', False, False, False,
       'sgd_text_sdt_separated_dialogue_all'),
      ('active_slots_slot_names', 'active', False, False, False,
       'sgd_text_sdt_separated_dialogue_active'),
      ('all_slots_slot_ids', 'all', True, False, False,
       'sgd_text_sdt_separated_dialogue_all_slot_ids'),
      ('all_slots_cat_val_mcq', 'all', False, True, False,
       'sgd_text_sdt_separated_dialogue_all_cat_val_mcq'),
      ('all_slots_d3st_mcq', 'all', False, True, True,
       'sgd_text_sdt_separated_dialogue_all_mcq_d3st'),
      ('all_slots_slot_ids_d3st_mcq', 'all', True, True, True,
       'sgd_text_sdt_separated_dialogue_all_slot_ids_mcq_d3st'))
  def test_generate_data(self, target_format, use_slot_ids, mcq_cat_vals,
                         use_intent_slot_descs, ref_output_filename):
    temp_output = os.path.join(self._tempdir, 'output')
    ref_output = os.path.join(self._testdata_dir, 'show_dont_tell',
                              ref_output_filename)

    with flagsaver.flagsaver(
        input_dir=os.path.join(self._testdata_dir, 'sgd_data'),
        output_path=temp_output,
        subdirs=['train'],
        prompt_format='separated',
        prompt_indices=['0'],
        context_format='dialogue',
        target_format=target_format,
        add_intents=False,
        use_slot_ids=use_slot_ids,
        randomize_slots=False,
        randomize_intents=False,
        mcq_cat_vals=mcq_cat_vals,
        mcq_intents=False,
        randomize_cat_vals=False,
        use_intent_slot_descs=use_intent_slot_descs,
    ):
      create_sgd_sdt_data.main([])

    with tf.io.gfile.GFile(temp_output) as temp_f, tf.io.gfile.GFile(
        ref_output) as ref_f:
      self.assertEqual(temp_f.readlines(), ref_f.readlines())

  def test_generate_sgdx_data(self):
    temp_output = os.path.join(self._tempdir, 'output')
    ref_output = os.path.join(self._testdata_dir, 'show_dont_tell',
                              'sgd_text_sdt_separated_dialogue_all_sgdx')

    with flagsaver.flagsaver(
        input_dir=os.path.join(self._testdata_dir, 'sgd_data'),
        output_path=temp_output,
        sgdx_dir=os.path.join(self._testdata_dir, 'sgdx_data'),
        subdirs=['train'],
        prompt_format='separated',
        prompt_indices=['0'],
        context_format='dialogue',
        target_format='all',
        add_intents=False,
        use_slot_ids=False,
        randomize_slots=False,
        randomize_intents=False,
        mcq_cat_vals=False,
        mcq_intents=False,
        randomize_cat_vals=False,
    ):
      create_sgd_sdt_data.main([])

    with tf.io.gfile.GFile(temp_output) as temp_f, tf.io.gfile.GFile(
        ref_output) as ref_f:
      self.assertEqual(temp_f.readlines(), ref_f.readlines())

  @parameterized.named_parameters(
      ('all_slots_slot_names_intent', False,
       'sgd_text_sdt_separated_dialogue_all_intent'),
      ('all_slots_slot_names_intent_mcq', True,
       'sgd_text_sdt_separated_dialogue_all_intent_mcq'))
  def test_generate_intent(self, mcq_intents, ref_output_filename):
    temp_output = os.path.join(self._tempdir, 'output')
    ref_output = os.path.join(self._testdata_dir, 'show_dont_tell',
                              ref_output_filename)

    with flagsaver.flagsaver(
        input_dir=os.path.join(self._testdata_dir, 'sgd_data'),
        output_path=temp_output,
        subdirs=['train'],
        prompt_format='separated',
        prompt_indices=['0'],
        context_format='dialogue',
        target_format='all',
        add_intents=True,
        use_slot_ids=False,
        randomize_slots=False,
        randomize_intents=False,
        mcq_cat_vals=False,
        mcq_intents=mcq_intents,
        randomize_cat_vals=False,
    ):
      create_sgd_sdt_data.main([])

    with tf.io.gfile.GFile(temp_output) as temp_f, tf.io.gfile.GFile(
        ref_output) as ref_f:
      self.assertEqual(temp_f.readlines(), ref_f.readlines())


if __name__ == '__main__':
  tf.test.main()
