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

"""Utils for creating text-to-text data."""

import dataclasses
import os
from typing import Dict, MutableSequence

from lingvo import compat as tf


# TODO(jeffreyzhao): Support extending with multiple fields
@dataclasses.dataclass
class TextToTextExample:
  """A single text-to-text dialogue example.

  Attributes:
    src: Input text for the model.
    tgt: Target text for the model.
    dialog_id: Id of dialog this example was generated from.
    turn: Turn of dialog this example was generated from.
    metadata: Any other key-value pairs to be included in the output TF Example.
    frame: Frame of the dialog this example was generated from.
  """
  src: str
  tgt: str
  dialog_id: str
  turn: int
  metadata: Dict[str, str] = dataclasses.field(default_factory=dict)
  frame: int = 0


def write_data(examples: MutableSequence[TextToTextExample],
               output_path: str) -> None:
  """Writes examples to the given output path.

  Args:
    examples: A list of formatted examples to write out
    output_path: The file path to write examples out to
  """

  def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

  def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

  tf.io.gfile.makedirs(os.path.dirname(output_path))

  with tf.io.TFRecordWriter(output_path) as out_file:
    for example in examples:
      features = {
          'input': _bytes_feature(example.src.encode('utf-8')),
          'value': _bytes_feature(example.tgt.encode('utf-8')),
          'dialog_id': _bytes_feature(example.dialog_id.encode('utf-8')),
          'turn': _int64_feature(example.turn)
      }
      for key, val in example.metadata.items():
        assert key not in ('input', 'value', 'dialog_id', 'turn')
        features[key] = _bytes_feature(val.encode('utf-8'))
      tf_example = tf.train.Example(
          features=tf.train.Features(feature=features))
      out_file.write(tf_example.SerializeToString())
    tf.logging.info('Wrote %s with %d examples', os.path.basename(output_path),
                    len(examples))


def decode_fn(record_bytes: tf.Tensor) -> Dict[str, tf.Tensor]:
  return tf.io.parse_single_example(
      record_bytes, {
          'input': tf.io.FixedLenFeature([], dtype=tf.string),
          'value': tf.io.FixedLenFeature([], dtype=tf.string),
          'dialog_id': tf.io.FixedLenFeature([], dtype=tf.string),
          'turn': tf.io.FixedLenFeature([], dtype=tf.int64)
      })
