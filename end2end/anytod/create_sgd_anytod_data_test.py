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

"""Tests for create_critic_data."""

import json
import os

from absl import flags
from absl.testing import parameterized
from task_oriented_dialogue.end2end.anytod import create_sgd_anytod_data
import tensorflow as tf

FLAGS = flags.FLAGS

TEST_DIR = 'zero_shot_task_oriented_dialog/testdata'


class CreateSgdAnytodDataTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self._testdata_dir = os.path.join(
        FLAGS.test_srcdir,
        TEST_DIR
    )

  def _load_testdata(self):
    dialogs = []
    with tf.io.gfile.GFile(
        os.path.join(self._testdata_dir, 'sgd_train_anytod.json')
    ) as f:
      js = json.load(f)
      for d in js:
        dialogs.append(d)
    with tf.io.gfile.GFile(
        os.path.join(self._testdata_dir, 'sgd_train_schema.json')
    ) as f:
      schema = json.load(f)
    return dialogs, schema

  def _read_tfrecords(self, filename):
    def _decode_value(v):
      v = v.numpy()
      if isinstance(v, bytes):
        v = v.decode('utf-8')
      return v

    feat_descs = {
        'input': tf.io.FixedLenFeature([], tf.string),
        'value': tf.io.FixedLenFeature([], tf.string),
        'dialog_id': tf.io.FixedLenFeature([], tf.string),
        'turn': tf.io.FixedLenFeature([], tf.int64),
        'frame': tf.io.FixedLenFeature([], tf.int64),
        'service': tf.io.FixedLenFeature([], tf.string),
        'metadata': tf.io.FixedLenFeature([], tf.string),
        'policy_table': tf.io.FixedLenFeature([], tf.string),
    }
    ds = tf.data.TFRecordDataset(filename)
    ds = ds.map(lambda r: tf.io.parse_single_example(r, feat_descs))
    exs = []
    for r in ds:
      exs.append({k: _decode_value(v) for k, v in r.items()})
    return exs

  @parameterized.named_parameters(
      ('basic', False, 'sgd_anytod_data_expected.tfrecord'),
      ('use_cat_slots', True, 'sgd_anytod_data_cat_slots_expected.tfrecord'),
  )
  def test_generate_data(self, use_cat_slots, expected_filename):
    dialogs, schema = self._load_testdata()
    converter = create_sgd_anytod_data.Converter(
        dialogs,
        schema,
        create_sgd_anytod_data.Mode.ZOMBIE_HISTORY_2PASS,
        False,
        use_cat_slots,
        False,
    )
    examples = converter.convert_to_anytodog_format()
    tempdir = self.create_tempdir().full_path
    with tf.io.TFRecordWriter(os.path.join(tempdir, 'actual.tfrecord')) as rw:
      for ex in examples:
        rw.write(ex.build_tf_example().SerializeToString())

    actual = self._read_tfrecords(os.path.join(tempdir, 'actual.tfrecord'))

    # For copying data.
    # print('ACTUAL:', os.path.join(tempdir, 'actual.tfrecord'))
    # print('EXPECTED:', os.path.join(self._testdata_dir, expected_filename))
    # __import__('pdb').set_trace()

    expected = self._read_tfrecords(
        os.path.join(self._testdata_dir, expected_filename)
    )

    for exp, act in zip(expected, actual):
      self.assertDictEqual(exp, act)


if __name__ == '__main__':
  tf.test.main()
