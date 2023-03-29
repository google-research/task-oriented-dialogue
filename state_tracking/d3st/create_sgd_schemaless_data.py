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

# pytype: skip-file
r"""Create text format SGD data for generative models.

This is version 2, which generates the schemaless data format:

0=[slot0 desc] 1=[slot1 desc]... i0=[intent0 desc] i1=[intent1 desc] \
[user] utterance [system] utterance... \t
[states] i=[value_i] j=[value_j]... \
[intents] ik il... \
[req_slots] m n...
"""
import collections
import copy
import dataclasses
import json
import os
import random
import string
from typing import Any, Dict, List, Tuple
from absl import app
from absl import flags

import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('sgd_file', None, 'The SGD json file path.')
flags.DEFINE_string('schema_file', None, 'Schema file path.')
flags.DEFINE_string('output_file', None, 'Output file path.')
flags.DEFINE_string(
    'delimiter', '=', 'Delimiter to separate slot/intent IDs '
    'from their descriptions or values.')
flags.DEFINE_enum('level', 'dst', ['dst', 'dst_intent', 'dst_intent_act'],
                  ('Which level of information should be '
                   'generated: '
                   'dst: Only generate slots for DST'
                   'dst_intent: Generate DST + intent (including user request)'
                   'dst_intent_act: Generate DST + intent + actions.'))
flags.DEFINE_enum('data_format', 'full_desc',
                  ['full_desc', 'item_name', 'rand_name'],
                  ('Format of the schemaless data:'
                   'full_desc: Use full language description as the item '
                   'description; '
                   'item_name: Use item name as the item description.'
                   'rand_name: Use random string as the item description.'))
flags.DEFINE_bool('lowercase', True, 'If True, lowercase everything.')
flags.DEFINE_bool('randomize_items', True, 'If True, randomize the order of '
                  'schema items.')
flags.DEFINE_enum(
    'multiple_choice', 'none', ('none', 'a', '1a'),
    'Whether to use multiple choice prompting for categorical slots.'
    'none: Don\'t use multiple choice prompting. '
    'a: Use the prompt "1: ... a) b) c)." '
    '1a: Use the prompt "1: ... 1a) 1b) 1c)."')
flags.DEFINE_float('data_percent', 0.0, 'If not 0, the percentage of data to '
                   'be generated.')
flags.DEFINE_bool(
    'uniform_domain_distribution', False, 'When data_percent > 0'
    ' make sure domains are (close-to) uniform distribution.')


@dataclasses.dataclass
class TurnInfo:
  """Information extracted from dialog turns."""
  out_ctx_str: str = ''
  out_ctx_with_desc_str: str = ''
  out_state_str: str = ''
  out_act_str: str = ''
  prev_state_str: str = ''
  delta_state_str: str = ''
  prev_sys_response: str = ''
  history_states: str = ''
  curr_user_utt: str = ''
  out_intent_str: str = ''
  new_states: str = ''
  new_intents: str = ''
  curr_utt: str = ''
  user_turn: bool = False
  turn_domain: str = ''
  dialogue_id: str = ''
  turn_id: str = ''
  frame_id: str = ''


def _merge_domain_slot(domain: str, slot_name: str):
  return f'{domain}-{slot_name}'


SchemaInfo = Dict[str, Dict]


def load_schema() -> Tuple[collections.OrderedDict, SchemaInfo]:
  """Loads schema items and descriptions.

  Returns:
    A tuple, including an ordered dictionary whose keys are slot names and
    values are placeholder values, and a dictionary whose keys are slot names
    and values are descriptions.
  """
  # We need to preserve state orders since we hope the model learns to generate
  # states in consistent order.
  # TODO(yuancao): We might need to do this for intents/actions as well (in case
  # multiple intents/actions turn out to be a problem).
  # TODO(jeffreyzhao): Clean up how we store schema information by using a
  # dataclass.
  slots = collections.OrderedDict()
  item_desc = {
      'slots': {},
      'intents': {},
      'is_categorical': {},
      'possible_values': {},
      'slots_rand_name': {},
      'intents_rand_name': {}
  }
  with tf.io.gfile.GFile(FLAGS.schema_file) as sm_file:
    for schema in json.load(sm_file):
      domain = schema['service_name']
      slots.update({
          _merge_domain_slot(domain, slot['name']): ''
          for slot in schema['slots']
      })
      item_desc['slots'].update({
          _merge_domain_slot(domain, slot['name']): slot['description']
          for slot in schema['slots']
      })

      for slot in schema['slots']:
        name = _merge_domain_slot(domain, slot['name'])
        is_cat = slot['is_categorical']
        poss_vals = slot['possible_values']

        # If this is a categorical slot but the possible value are all numeric,
        # consider this as a noncat slot.
        if is_cat and all([v.isdigit() for v in poss_vals]):
          poss_vals = []
          is_cat = False

        item_desc['is_categorical'][name] = is_cat
        item_desc['possible_values'][name] = poss_vals

      item_desc['intents'].update({
          _merge_domain_slot(domain, intent['name']): intent['description']
          for intent in schema['intents']
      })

      if FLAGS.data_format == 'rand_name':
        item_desc['slots_rand_name'].update({
            _merge_domain_slot(domain, slot['name']):
            ''.join(random.sample(list(slot['name']), len(slot['name'])))
            for slot in schema['slots']
        })
        # pylint: disable=g-complex-comprehension
        item_desc['intents_rand_name'].update({
            _merge_domain_slot(domain, intent['name']):
            ''.join(random.sample(list(intent['name']), len(intent['name'])))
            for intent in schema['intents']
        })
        # pylint: enable=g-complex-comprehension
  return slots, item_desc


def _process_user_turn(state: Dict[str, Any], turn_info: TurnInfo,
                       cumu_slots: collections.OrderedDict, domain: str,
                       item_desc: SchemaInfo,
                       state_dict: Dict[str, List[str]]) -> Dict[str, int]:
  """Updates turn_info and cumu_slots based on user turn input.

  Args:
    state: A dictionary containing state info.
    turn_info: A TurnInfo object accmulating essential info from each turn.
    cumu_slots: An OrderedDict containing cmumulative slot information.
    domain: A string, domain (service) of the turn.
    item_desc: A dictionary of items and their descriptions.
    state_dict: A dictionary of states from the current turn.

  Returns:
    A dictionary that maps slot descriptions to ids.
  """
  slot_values = state['slot_values']
  domain_slot_values = {}
  for slot, value in slot_values.items():
    domain_slot_values[_merge_domain_slot(domain, slot)] = value
  slot_values = domain_slot_values

  # Order of slots is preserved. Meanwhile new values of the same
  # slots will overwrite existing ones.
  for slot, value in slot_values.items():
    if slot not in cumu_slots:
      raise ValueError(f'Unknown slot: {slot}.')
    cumu_slots.update({slot: ' | '.join(value)})

  # Clean up.
  desc_to_slot_id = {}
  slots = list(item_desc['slots'].keys())
  if FLAGS.randomize_items:
    random.shuffle(slots)
  # In multi-domain turn case, desc_prefix already contains desc from the
  # previous domain.
  slot_id = len(state_dict['slot_desc'])
  for slot in slots:
    if FLAGS.data_format == 'full_desc':
      desc = item_desc['slots'][slot]
    elif FLAGS.data_format == 'item_name':
      desc = slot
    elif FLAGS.data_format == 'rand_name':
      desc = item_desc['slots_rand_name'][slot]

    # If we are generating with multiple choice, append this prompt.
    if FLAGS.multiple_choice != 'none' and item_desc['is_categorical'][slot]:
      possible_values = item_desc['possible_values'][slot]
      if FLAGS.randomize_items:
        random.shuffle(possible_values)
      assert len(possible_values) < len(string.ascii_lowercase)
      letters = list(string.ascii_lowercase)

      possible_values_pieces = []
      for letter, value in zip(letters, possible_values):
        if FLAGS.multiple_choice == '1a':
          possible_values_pieces.append(f'{slot_id}{letter}) {value}')
        elif FLAGS.multiple_choice == 'a':
          possible_values_pieces.append(f'{letter}) {value}')
      desc += ' ' + ' '.join(possible_values_pieces)

    # Only consider slots in the utterance domain.
    if domain in slot.split('-')[0]:
      # Description prefix to be included in each turn.
      t = f' {slot_id}{FLAGS.delimiter}'
      desc_to_slot_id[slot] = slot_id
      state_dict['slot_desc'].append(t + desc.lower() +
                                     ' ' if FLAGS.lowercase else t + desc + ' ')

      state_str = ''
      # Corresponding values for active slots.
      if cumu_slots[slot]:
        value = cumu_slots[slot]
        if (FLAGS.multiple_choice != 'none' and
            item_desc['is_categorical'][slot] and value != 'dontcare'):
          # Convert to multiple choice for categorical slots.
          assert value in possible_values
          state_str = t + str(slot_id) + letters[possible_values.index(value)]
        else:
          state_str = t + value

      turn_info.out_state_str += (
          state_str.lower() if FLAGS.lowercase else state_str)
      turn_info.turn_domain = domain
      slot_id += 1

  # Handle intents.
  # In multi-domain turn case, intent list already contains intents from the
  # previous domain.
  intents = list(item_desc['intents'].keys())
  if FLAGS.randomize_items:
    random.shuffle(intents)
  intent_id = len(state_dict['intent_desc'])
  for intent in intents:
    if FLAGS.data_format == 'full_desc':
      desc = item_desc['intents'][intent]
    if FLAGS.data_format == 'item_name':
      desc = intent
    elif FLAGS.data_format == 'rand_name':
      desc = item_desc['intents_rand_name'][intent]

    # Only consider slots in the utterance domain.
    if domain in intent:
      active_intent = domain + '-' + state['active_intent']
      # Description prefix to be included in each turn.
      t = f' i{intent_id}{FLAGS.delimiter}'
      intent_str = ''
      if active_intent == intent:
        intent_str = ' ' + t[:-1]

      state_dict['intent_desc'].append(t + desc.lower() +
                                       ' ' if FLAGS.lowercase else t + desc +
                                       ' ')
      if intent_str:
        state_dict['intent_ids'].append(intent_str)
      intent_id += 1

  # Handle requested slots.
  for req_slot in state['requested_slots']:
    slot_name = domain + '-' + req_slot
    assert slot_name in desc_to_slot_id, (
        'Requested slots must be in the slot list!')
    req_slot_id = desc_to_slot_id[slot_name]
    # Note the order of requested slots is totally determined by the user's
    # utterance, and is not guaranteed to be sorted.
    state_dict['req_slots'].append(str(req_slot_id))

  return desc_to_slot_id


def _process_agent_turn(actions: List[Dict[str, Any]], turn_info: TurnInfo,
                        domain: str, desc_to_slot_id: Dict[str, int]) -> None:
  """Updates turn_info based on the system actions.

  Args:
    actions: A list of strings for system actions.
    turn_info: A Turninfo object accmulating essential info from each turn.
    domain: A string, domain (service) of the current turn.
    desc_to_slot_id: A dictionary that maps descriptions to slot ids.
  """
  turn_info.prev_state_str = turn_info.out_state_str
  turn_info.out_act_str += ' [actions] '
  acts = {}
  for action in actions:
    act = action['act']
    slot = action['slot']
    # Note that we don't include api function values but only names, as these
    # values are supposed to be delexicalized and retrieved from db.
    # values = action['values']
    if act not in acts:
      acts[act] = ''
    if slot:
      act_slot = _merge_domain_slot(domain, slot)
      if act_slot in desc_to_slot_id:
        slot_id = desc_to_slot_id[act_slot]
        acts[act] += str(slot_id) + ';'
    else:
      acts[act] += 'none;'

  turn_info.out_act_str += ' '.join(
      [f'{action}({params})' for action, params in acts.items()])
  if FLAGS.lowercase:
    turn_info.out_act_str = turn_info.out_act_str.lower()


def process_turn(turn: Dict[str, Any], turn_info: TurnInfo,
                 cumu_slots: collections.OrderedDict, item_desc: SchemaInfo,
                 prefix: str, turn_id: int) -> Tuple[str, List[TurnInfo]]:
  """Collects information from a single turn.

  Args:
    turn: A dictionary containing original turn structure.
    turn_info: A dictionary accmulating essential info from each turn.
    cumu_slots: An OrderedDict containing cumumulative slot information.
    item_desc: A dictionary of scheam items and their descriptions.
    prefix: A string of the schema item description prefix.
    turn_id: Integer index of turn in dialogue.

  Returns:
    Prefix string (item descriptions) from the current turn and per-frame
    TurnInfo objects.
  """
  speaker = turn['speaker'].lower()
  user_turn = speaker == 'user'
  turn_info.user_turn = user_turn
  utt = turn['utterance']
  turn_info.curr_utt = f'[{speaker}] {utt} '
  turn_info.out_ctx_str += f'[{speaker}] {utt} '
  turn_info.turn_id = str(turn_id)
  if FLAGS.lowercase:
    turn_info.curr_utt = turn_info.curr_utt.lower()
    turn_info.out_ctx_str = turn_info.out_ctx_str.lower()
  # Intent and act strings are not accumulative.
  turn_info.out_act_str = ''
  if user_turn:
    turn_info.out_state_str = '[states]'
    turn_info.out_intent_str = '[intents]'

  desc_to_slot_id = {}
  turn_info_per_frame = []
  for frame_id, frames in enumerate(turn['frames']):
    domain = frames['service']
    turn_info.frame_id = str(frame_id)
    state_dict = {
        'slot_desc': [],
        'intent_desc': [],
        'intent_ids': [],
        'req_slots': []
    }

    if user_turn:
      # Multi-service turns are possible, each frame corresponds to one
      # service (domain).

      # Note: frames['slots'] is not used for generation.
      turn_info.out_state_str = '[states]'
      turn_info.out_intent_str = '[intents]'
      desc_to_slot_id = _process_user_turn(frames['state'], turn_info,
                                           cumu_slots, domain, item_desc,
                                           state_dict)
    else:
      _process_agent_turn(frames['actions'], turn_info, domain, desc_to_slot_id)

    # Add item description prefixes and states to outputs (coming from user
    # turns).
    user_turn_prefix = ''.join(state_dict['slot_desc'] +
                               state_dict['intent_desc'])
    if user_turn:
      turn_info.out_ctx_with_desc_str = user_turn_prefix + turn_info.out_ctx_str
    else:
      # Prefix from the previous user turn.
      turn_info.out_ctx_with_desc_str = prefix + turn_info.out_ctx_str
    turn_info.out_intent_str += ''.join(state_dict['intent_ids'])
    turn_info.out_intent_str += ' [req_slots] '
    turn_info.out_intent_str += ' '.join(state_dict['req_slots'])
    turn_info_per_frame.append(copy.deepcopy(turn_info))

  return user_turn_prefix, turn_info_per_frame


def write_examples(turn_list: List[TurnInfo], out_file: tf.io.gfile.GFile) -> None:
  """Format output example strings and write to file.

  Args:
    turn_list: A list of dict accmulating essential info from each turn.
    out_file: A GFile object for file output.
  """
  for turn_info in turn_list:
    # Write samples to file. Each example is divided into two parts
    # separated by \t, the first part being inputs to the model, and the
    # second part are labels for prediction.
    src = turn_info.out_ctx_with_desc_str
    tgt = ''

    if FLAGS.level == 'dst':
      if turn_info.user_turn:
        # Only output at user turns.
        tgt = turn_info.out_state_str
    elif FLAGS.level == 'dst_intent':
      if turn_info.user_turn:
        tgt = ' '.join([turn_info.out_state_str, turn_info.out_intent_str])
    elif FLAGS.level == 'dst_intent_act':
      if not turn_info.user_turn:
        # Only output at system turns, including:
        # state + action + responses
        turn_info.curr_utt = turn_info.curr_utt.replace('[system]',
                                                        '[response]')
        tgt = ' '.join([
            turn_info.out_state_str, turn_info.out_intent_str,
            turn_info.out_act_str, turn_info.curr_utt
        ])

    if tgt:
      # Add dialogue ID, turn ID and frame ID to the target for later eval.
      # Occasionally some examples include newline in the middle.
      example = (f'{" ".join(src.split())} \t{" ".join(tgt.split())}\t' +
                 f'{turn_info.dialogue_id}\t{turn_info.turn_id}\t' +
                 f'{turn_info.frame_id}')
      if FLAGS.lowercase:
        example = example.lower()
      print(example)
      out_file.write('{}\n'.format(example.strip()))


def example_filter(turn_list: List[TurnInfo]):
  """Extract specified percentage of examples.

  And ensure uniform domain distribution if specified.

  Args:
    turn_list: A list of TurnInfo containing all examples.

  Returns:
    Specified percentage of examples, with uniform domain distribution if
    needed.
  """
  if FLAGS.data_percent == 0.0:
    return turn_list

  out_sample_num = int(len(turn_list) * FLAGS.data_percent)
  if not FLAGS.uniform_domain_distribution:
    if FLAGS.randomize_items:
      random.shuffle(turn_list)
    return turn_list[:out_sample_num]
  else:
    domain_examples = {}
    domain_id = {}
    domain_count = 0
    for turn in turn_list:
      if turn.turn_domain in domain_id:
        domain_examples[domain_id[turn.turn_domain]].append(turn)
      else:
        domain_examples[domain_count] = [turn]
        domain_id[turn.turn_domain] = domain_count
        domain_count += 1

    # How many examples from each domain has been added to the final list.
    consumed_examples = {d: 0 for d in range(domain_count)}
    uniform_turn_list = []
    for s in range(out_sample_num):
      # Find first domain that still has unused examples.
      domain_id = s % domain_count
      for d in range(domain_count):
        cand_domain = (domain_id + d) % domain_count
        if len(domain_examples[cand_domain]) > consumed_examples[cand_domain]:
          domain_id = cand_domain
          break

      uniform_turn_list.append(
          domain_examples[domain_id][consumed_examples[domain_id]])
      consumed_examples[domain_id] += 1

    if FLAGS.randomize_items:
      random.shuffle(uniform_turn_list)

    return uniform_turn_list


def generate_data(ordered_slots, item_desc):
  """Generate SGD examples in text format.

  Args:
    ordered_slots: An ordered dictionary containing slot names.
    item_desc: A dictionary containing items and their descriptions.
  """
  if not tf.io.gfile.isdir(os.path.dirname(FLAGS.output_file)):
    tf.io.gfile.makedirs(os.path.dirname(FLAGS.output_file))
  with tf.io.gfile.GFile(FLAGS.output_file, 'w') as out_file:
    all_turns_per_frame = []
    for sgd_file in tf.io.gfile.glob(FLAGS.sgd_file):
      with tf.io.gfile.GFile(sgd_file) as sgd_in:
        for dlg in json.load(sgd_in):
          # Cumulative states throughout this dialog.
          cumu_slots = copy.deepcopy(ordered_slots)
          turn_info = TurnInfo()
          turn_info.dialogue_id = dlg['dialogue_id']
          prefix = ''
          for turn_idx, turn in enumerate(dlg['turns']):
            prefix, per_frame_turn_info = process_turn(turn, turn_info,
                                                       cumu_slots, item_desc,
                                                       prefix, turn_idx)
            all_turns_per_frame.extend(per_frame_turn_info)

        write_examples(example_filter(all_turns_per_frame), out_file)


def main(_):
  slots, item_desc = load_schema()
  generate_data(slots, item_desc)


if __name__ == '__main__':
  flags.mark_flag_as_required('sgd_file')
  flags.mark_flag_as_required('schema_file')
  flags.mark_flag_as_required('output_file')
  app.run(main)
