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

r"""Create STARv2 data for labeling by D3ST.

To generate exemplars to be hand-labeled:
blaze-local run -c opt \
  //learning/brain/research/babelfish/dialog/tools/starv2:create_star_d3st_data
  \
  -- --num_exs_per_task=5 --use_cat_slots --include_target_string=true

To generate examples for all of the STAR dataset:
blaze-local run -c opt \
  //learning/brain/research/babelfish/dialog/tools/starv2:create_star_d3st_data
  \
  -- --num_exs_per_task=1000000 --use_cat_slots --include_target_string=false
"""
import collections
import dataclasses
import difflib
import enum
import os
import random
import string
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from task_oriented_dialogue.end2end.anytod import starv2_lib
import tensorflow as tf


class Mode(enum.Enum):
  HANDLABEL = 'handlabel'  # Output text format for manual labelling.
  PREDICT = 'predict'  # Output tfrecords for prediction from D3ST.


_DATADIR = flags.DEFINE_string(
    'input_dir',
    '',
    'Input STAR datadir.',
)
_USE_CATEGORICAL = flags.DEFINE_boolean(
    'use_cat_slots', True, 'Whether to use categorical slots,'
)
_INCLUDE_TARGET = flags.DEFINE_boolean(
    'tgt_str', True, 'Whether to include target string.'
)
_NUM_EXS_PER_TASK = flags.DEFINE_integer(
    'num_exs_per_task', 1_000_000, 'Number of examples to take from each task.'
)
_OUTDIR = flags.DEFINE_string('output_dir', None, 'Output directory.')
_MODE = flags.DEFINE_enum_class('mode', Mode.HANDLABEL, Mode, 'Mode.')

# "intents" for each task
TASK_DESCS = {
    'apartment_schedule': 'user wants to schedule apartment viewing',
    'apartment_search': 'user wants to schedule apartment viewing',
    'bank_balance': 'user needs bank balance',
    'bank_fraud_report': 'user needs to file bank fraud report',
    'doctor_followup': 'user is asking about doctor followup',
    'doctor_schedule': 'user is asking about doctor schedule',
    'hotel_book': 'user wants to book hotel',
    'hotel_search': 'user wants to find a hotel',
    'hotel_service_request': 'user needs hotel service',
    'meeting_schedule': 'user wants to schedule a meeting',
    'party_plan': 'user wants to plan a party',
    'party_rsvp': 'user wants to make party rsvp',
    'plane_book': 'user wants to book a plane',
    'plane_search': 'user wants to book a plane',
    'restaurant_book': 'user wants to book a restaurant',
    'restaurant_search': 'user wants to search for a restaurant',
    'ride_book': 'user wants to book a rideshare',
    'ride_change': 'user wants to change their rideshare',
    'ride_status': 'user wants to find the status of their rideshare',
    'spaceship_access_codes': (
        'user needs to get into the combat deck on a spaceship'
    ),
    'spaceship_life_support': 'user needs to fix spaceship life support',
    'trip_directions': 'user needs trip directions',
    'trivia': 'user wants to play trivia',
    'weather': 'user wants to find weather',
}


@dataclasses.dataclass
class Example:
  """STAR D3ST example dataclass."""

  dialog_id: int
  turn: int
  inp: str
  tgt: str
  slot_ord: list[str]

  @property
  def slot_ord_str(self) -> str:
    return ', '.join(self.slot_ord)

  def build_tf_example(self) -> tf.train.Example:
    """Converts this Example dataclass into a tf.Example."""

    def _bytes_feature(value):
      """Returns a bytes_list from a string / byte."""
      return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
      """Returns an int64_list from a bool / enum / int / uint."""
      return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'input': _bytes_feature(self.inp.encode('utf-8')),
                'value': _bytes_feature(self.tgt.encode('utf-8')),
                'dialog_id': _bytes_feature(
                    str(self.dialog_id).encode('utf-8')
                ),
                'turn': _int64_feature(self.turn),
                'slot_ordering': _bytes_feature(
                    self.slot_ord_str.encode('utf-8')
                ),
            }
        )
    )


def generate_star_d3st_examples(
    data: starv2_lib.StarData,
) -> dict[str, list[Example]]:
  """Main data generation function."""

  # TODO(jeffreyzhao): Refactor starv2_lib to only do json access.

  def _clean_value(value):
    """Clean a value within a query belief state."""
    while True:
      if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]
      elif value.startswith('api.is_equal_to('):
        value = value[len('api.is_equal_to(') : -len(')')]
      else:
        return value.strip()

  def _is_categorical_slot(param):
    if not _USE_CATEGORICAL.value:
      return False
    return param.type == 'Categorical' and len(param.categories) < 10

  def _extract_wizard_task_info(dialog):
    """Extract information provided by info provided to wizard."""
    wizard_task = dialog.json['Scenario']['WizardTask']
    ret = []
    for line in wizard_task.split('\n'):
      line = line.strip()
      if line.startswith('*'):
        line = line.removeprefix('* ')
        if (
            line
            == 'You are the customer service AI of the international '
            'banking system'
        ):
          continue
        line = line.removeprefix('At the beginning of the conversation, ')
        line = line.removeprefix('At the beginning of the conversation')
        line = line.strip()
        ret.append(line)
    return ret

  # task -> examples for that task
  task_examples = collections.defaultdict(list)

  for task, task_desc in TASK_DESCS.items():
    api = data.task_to_api[task]
    for dialog_id in data.task_to_ids[task][: _NUM_EXS_PER_TASK.value]:
      dialog = data.dialogs[dialog_id]
      dialog_json = dialog.json
      convo_hist = []
      for turn, event in enumerate(dialog_json['Events']):
        # Add to conversation history.
        if starv2_lib.is_user_event(event) or starv2_lib.is_wizard_event(event):
          speaker = 'user' if event['Agent'] == 'User' else 'system'
          text = event['Text']
          # Some STAR dialogues have necessary info in the wizard task info
          if not convo_hist:
            wiz_info = _extract_wizard_task_info(dialog)
            text = '. '.join(wiz_info + [text])
          convo_hist.append((speaker, text))

        if starv2_lib.is_user_event(event):
          name_to_param = {p.name: p for p in api.params}
          slot_ord = starv2_lib.Ordering([p.name for p in api.params], True)
          cat_value_ord = {}
          for p in api.params:
            if _is_categorical_slot(p):
              cat_value_ord[p.name] = starv2_lib.Ordering(p.categories, True)

          # Construct D3ST prompt.
          prompt = []
          for slot_ind, slot in slot_ord:
            param = name_to_param[slot]
            slot_desc = param.readable_name
            pieces = [f'{slot_ind}={slot_desc}']
            if _is_categorical_slot(param):
              for val_ind, val in cat_value_ord[slot]:
                val_let = string.ascii_letters[val_ind]
                pieces.append(f'{slot_ind}{val_let}) {val}')
            prompt.append(' '.join(pieces))
          prompt.append(f'i0={task_desc}')
          prompt = ' '.join(prompt)

          # Get the query belief state.
          # This query belief state is a few turns in the future and likely low
          # quality, but helps with hand labeling.
          _, query_event = dialog.get_closest_event(
              turn, starv2_lib.is_query_event
          )
          if not query_event:
            _, query_event = dialog.get_closest_event(
                turn, starv2_lib.is_query_event, reverse=True
            )
          if query_event is None:
            next_bs = {}
          else:
            next_bs = starv2_lib.get_query_event_belief_state(query_event)
          next_bs = {slot: _clean_value(val) for slot, val in next_bs.items()}

          # Construct a target string.
          tgt = []
          if _INCLUDE_TARGET.value:
            for slot_ind, slot in slot_ord:
              param = name_to_param[slot]
              val = next_bs.get(slot, None)
              if val is None:
                # If this is a bank domain "forgot" slot, have this be the value
                if task in ('bank_balance', 'bank_fraud_report') and slot in (
                    'PIN',
                    'AccountNumber',
                ):
                  val = 'forgot'
                elif param.required:
                  logging.warning('%s not in future belief state!!', val)
                  continue
                else:
                  continue
              # Categorical values.
              if _is_categorical_slot(param):
                if val not in cat_value_ord[slot]:
                  old_val = val
                  # Find a close value from the list of possible values.
                  matches = difflib.get_close_matches(val, param.categories)
                  if not matches:
                    continue
                  val = matches[0]
                  logging.warning(
                      '%s not in categories, using %s', old_val, val
                  )
                val_ind = cat_value_ord[slot].get_idx(val)
                val_let = string.ascii_letters[val_ind]
                tgt.append(f'{slot_ind}={slot_ind}{val_let}')
              else:
                tgt.append(f'{slot_ind}={val}')

          convo_str = ' '.join(
              f'[{speaker}] {utt}' for speaker, utt in convo_hist
          )
          inp = f'{prompt} {convo_str}'
          tgt = ' '.join(tgt)
          tgt = f'[states] {tgt} [intents] i0 [req_slots]'
          task_examples[task].append(
              Example(dialog_id, turn, inp, tgt, slot_ord.tolist())
          )

  return task_examples


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  random.seed(123)
  options = starv2_lib.Options(
      starv2_lib.ExampleFormat.TRANSITIONS_ANYTOD, False, False
  )
  starv2_lib.set_star_version(starv2_lib.StarVersion.V1)
  data = starv2_lib.load_star_jsons(_DATADIR.value, options)
  task_examples = generate_star_d3st_examples(data)

  if _MODE.value == Mode.HANDLABEL:
    for task, exs in task_examples.items():
      task_fname = os.path.join(_OUTDIR.value, f'{task}_handlabel.txt')
      with tf.io.gfile.GFile(task_fname, 'w') as f:
        for ex in exs:
          f.write(str(ex.dialog_id))
          f.write(ex.inp.lower().replace('\n', ' '))
          f.write(ex.tgt.lower().replace('\n', ' '))
          f.write(ex.slot_ord_str)
      logging.info('Wrote %d examples to %s', len(exs), task_fname)
  elif _MODE.value == Mode.PREDICT:
    all_tfexs = []
    for task, exs in task_examples.items():
      for ex in exs:
        all_tfexs.append(ex.build_tf_example())
    fname = os.path.join(_OUTDIR.value, 'all_exs_predict.tfrecord')
    with tf.io.TFRecordWriter(fname) as rw:
      for tfex in all_tfexs:
        rw.write(tfex.SerializeToString())
    logging.info('Wrote %d examples to %s', len(all_tfexs), fname)


if __name__ == '__main__':
  app.run(main)
