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

r"""Converts T5X predictions on SGD to DSTC8 official format for evaluation."""

import collections
import json
import os
import re
from typing import Dict, Optional, Sequence, Union

from absl import app
from absl import flags
from absl import logging
from state_tracking.utils import sgd_utils
import tensorflow as tf


_T5X_PREDICTIONS_JSONL = flags.DEFINE_string(
    't5x_predictions_jsonl', None,
    'Input JSONL file with T5X model predictions.')
_DSTC8_DATA_DIR = flags.DEFINE_string(
    'dstc8_data_dir', None,
    'Directory for the downloaded DSTC8 data, which contains '
    'the dialogue files and schema files of all datasets (train, dev, test)')
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', None,
    'Output directory for JSON-format model predictions for official DSTC8 '
    'evaluation.')
_DATASET_SPLIT = flags.DEFINE_enum('dataset_split', 'test',
                                   ['train', 'dev', 'test'],
                                   'Dataset split for evaluation.')
_DELIMITER = flags.DEFINE_string(
    'delimiter', '=', 'Delimiter to separate '
    'slot/intent IDs from their descriptions or '
    'values.')

_SDT_CAT_SLOT_IDENTIFIER = 'of possible values'


def _create_categorical_slot_to_value_map(
    input_str: str) -> Dict[str, Dict[str, str]]:
  """Creates mappings from letters to values for categorical slots."""
  slot_values = input_str.split('[slots]')[1].split('[context]')[0].strip()
  slot_to_option_to_value = collections.defaultdict(dict)
  for slot, value in re.findall(
      rf'(\w+){_DELIMITER.value}(.*?)(?=\w+{_DELIMITER.value}|$)', slot_values):
    if _SDT_CAT_SLOT_IDENTIFIER not in value:
      continue
    options_str = value.split(_SDT_CAT_SLOT_IDENTIFIER)[1].strip()
    for option, option_value in re.findall(r'([a-z])\) (.*?)(?=[a-z]\)|$)',
                                           options_str):
      slot_to_option_to_value[slot][option] = option_value.strip()

  return slot_to_option_to_value


def _normalize_value_prediction(
    slot_name: str, value: str,
    slot_to_option_to_value: Dict[str, Dict[str, str]]) -> Optional[str]:
  """Normalizes a predicted value and maps a categorical option to value."""
  value = value.strip()
  if value == 'none':
    value = None

  # Map decoded multiple choice letters back to actual value for cat slots.
  elif slot_name in slot_to_option_to_value:
    if value in slot_to_option_to_value[slot_name]:
      value = slot_to_option_to_value[slot_name][value]
    # Print cases where model didn't decode a valid multiple choice letter.
    elif value != 'dontcare':
      logging.info(
          'Unexpected slot scenario. slot_name %s. value %s. '
          'slot_to_option_to_value %s', slot_name, value,
          slot_to_option_to_value)

  return value


def populate_json_predictions(
    dialog_id_to_dialogue: Dict[str, sgd_utils.DialoguesDict],
    frame_predictions: Dict[str, Union[str, Dict[str, str]]]) -> None:
  """Populates a dialogue JSON dictionary with frame-level T5X model outputs.

  Given a single prediction from frame_predictions, this looks up the
  corresponding frame from dialog_id_to_dialogue and modifies it in-place by
  inserting the predictions into the dialogue state field.

  Args:
    dialog_id_to_dialogue: A mapping from dialog id to the dialogue json object
    frame_predictions: A dict containing T5X predictions and example metadata
  """
  preds = frame_predictions['prediction']
  if not isinstance(preds, str):
    raise ValueError(f"'preds' must be string type, "
                     f'not {type(preds)}. preds: {preds}')
  dialog_id = frame_predictions['input']['dialogue_id']
  turn_id = int(frame_predictions['input']['turn_id'])
  frame_id = int(frame_predictions['input']['frame_id'])

  if dialog_id not in dialog_id_to_dialogue:
    raise ValueError(f'Dialogue ID {dialog_id} not found.')

  frame = dialog_id_to_dialogue[dialog_id]['turns'][turn_id]['frames'][frame_id]

  input_str = frame_predictions['input']['inputs_pretokenized']

  # Create a dict(slot -> dict(multiple-choice letter -> value)) for cat slots.
  slot_to_option_to_value = _create_categorical_slot_to_value_map(input_str)

  # Read and populate all slot value predictions.
  # TODO(harrisonlee): Support intents and requested slots.
  slot_preds = preds.split('[state]')[1]
  for slot_name, value in re.findall(
      rf'(\w+){_DELIMITER.value}(.*?)(?=\w+{_DELIMITER.value}|$)', slot_preds):
    value = _normalize_value_prediction(slot_name, value,
                                        slot_to_option_to_value)
    if value:
      frame['state']['slot_values'][slot_name] = [value]


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Load dialogues and flatten into dict(dialogue_id->dialogue).
  subdir_to_dialogues = {}
  sgd_utils.load_dialogues_to_dict(_DSTC8_DATA_DIR.value, _DATASET_SPLIT.value,
                                   subdir_to_dialogues)
  dialog_id_to_dialogue = {}
  for dialogues in subdir_to_dialogues[_DATASET_SPLIT.value].values():
    for dialog in dialogues:
      dialog_id_to_dialogue[dialog['dialogue_id']] = dialog

  # Erase ground truth state values.
  for dial in dialog_id_to_dialogue.values():
    for turn in dial['turns']:
      for frame in turn['frames']:
        if 'state' in frame:
          frame['state']['slot_values'] = {}
          frame['state']['requested_slots'] = []
          frame['state']['active_intent'] = 'NONE'

  # Read JSONL predictions.
  with tf.io.gfile.GFile(_T5X_PREDICTIONS_JSONL.value, 'r') as predictions_file:
    for line in predictions_file:
      frame_predictions = json.loads(line)
      populate_json_predictions(dialog_id_to_dialogue, frame_predictions)

  # Write JSON predictions.
  output_dir = _OUTPUT_DIR.value
  if not tf.io.gfile.isdir(output_dir):
    tf.io.gfile.makedirs(output_dir)

  with tf.io.gfile.GFile(os.path.join(output_dir, 'dialogues_all.json'),
                         'w') as output_file:
    json.dump(
        list(dialog_id_to_dialogue.values()),
        output_file,
        indent=2,
        separators=(',', ': '))


if __name__ == '__main__':
  flags.mark_flag_as_required('t5x_predictions_jsonl')
  flags.mark_flag_as_required('dstc8_data_dir')
  flags.mark_flag_as_required('output_dir')
  app.run(main)
