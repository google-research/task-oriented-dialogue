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

"""Create MultiWOZ schemaless training data for T5x models.

For processing TRADE pre-processed data,
see create_multiwoz21_trade_schemaless_data.py.
"""

import dataclasses
import os
import random
import string
from typing import Dict, List, Set

from absl import flags
from state_tracking.utils import multiwoz_utils
from state_tracking.utils import text_to_text_utils
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('multiwoz_dir', None,
                    'Required. Path to the original MultiWOZ datasets.')
flags.DEFINE_string('output_dir', None, 'Required. Output file path.')
flags.DEFINE_string('schema_file', None,
                    'Required. MultiWOZ schema file in 2.2/SGD format.')
flags.DEFINE_enum('multiwoz_version', '2.4', ('2.1', '2.2', '2.3', '2.4'),
                  'Required. MultiWOZ dataset version.')
flags.DEFINE_integer('random_seed', None,
                     'Random seed. If None, random is not seeded.')
flags.DEFINE_enum(
    'description_type', 'full_desc',
    ('full_desc', 'full_desc_with_domain', 'item_name', 'shuffled_item_name'),
    'What to use for the slot descriptions. '
    'full_desc: A natural language description. '
    'full_desc_with_domain: Domain, followed by natural language description. '
    'item_name: The name of the slot. '
    'shuffled_item_name: Random permutation of the slot name.')
flags.DEFINE_string('delimiter', ':',
                    'Delimiter between id and slot description.')
flags.DEFINE_enum(
    'multiple_choice', 'none', ('none', 'a', '1a'),
    'Whether to use multiple choice prompting for categorical slots.'
    'none: Don\'t use multiple choice prompting. '
    'a: Use the prompt "1: ... a) b) c)." '
    '1a: Use the prompt "1: ... 1a) 1b) 1c)."')
flags.DEFINE_bool(
    'use_active_domains_only', False,
    'If true, only include domains that are active in this dialogue.')
flags.DEFINE_list(
    'blocked_domains', [], 'Don\'t include these domains '
    'if set. This is used to run zero-shot '
    'cross-domain experiments as in paper '
    'https://aclanthology.org/2021.naacl-main.448.pdf.')

Json = multiwoz_utils.Json
SchemaInfo = multiwoz_utils.SchemaInfo
TextToTextExample = text_to_text_utils.TextToTextExample


@dataclasses.dataclass
class Options:
  multiwoz_version: str
  description_type: str
  delimiter: str
  multiple_choice: str
  use_active_domains_only: bool
  blocked_domains: Set[str]


def create_schemaless_data(dialogs_by_id: Dict[str, multiwoz_utils.MultiwozDialog],
                           schema_info: SchemaInfo,
                           slot_descriptions: Dict[str, List[str]],
                           options: Options) -> List[TextToTextExample]:
  """Converts raw MultiWOZ data into schemaless examples."""

  def _multiple_choice_answer(slot_id: int, letters: List[str],
                              possible_values_shuffled: List[str], value: str):
    """Get answer for multiple choice prompt."""
    if value == 'none':
      return 'none'
    if value == 'dontcare':
      return 'dontcare'
    # Often we have have "guest house" when the categorical
    # value is "guesthouse".
    if value == 'guest house':
      value = 'guesthouse'

    if value not in possible_values_shuffled:
      tf.logging.warning('Value "%s" not in possible values %s', value,
                         possible_values_shuffled)
      value_nospaces = value.replace(' ', '')
      if value_nospaces in possible_values_shuffled:
        letter = letters[possible_values_shuffled.index(value_nospaces)]
      else:
        # Give up and return unknown as the value.
        tf.logging.warning('Value "%s" not in possible values %s', value,
                           possible_values_shuffled)
        return 'unknown'
    else:
      letter = letters[possible_values_shuffled.index(value)]

    if options.multiple_choice == '1a':
      return f'{slot_id}{letter}'
    elif options.multiple_choice == 'a':
      return letter

  def _process_one_turn(
      dialog_id: str, turn: int, belief_state: Dict[str, str], history_str: str,
      active_domains: Set[str],
      slot_descriptions: Dict[str, List[str]]) -> TextToTextExample:
    """Creates a `TextToTextExample` from a turn in the dialogue."""
    # Generate a random mapping from slot name to index.
    # slot_names[i] will translate to "i:slot_names[i]".
    slot_names = list(slot_descriptions.keys())
    if options.use_active_domains_only:
      slot_names = list(
          filter(lambda name: multiwoz_utils.get_domain(name) in active_domains,
                 slot_names))
    random.shuffle(slot_names)

    prefix_pieces = []
    state_pieces = []
    for i, slot_name in enumerate(slot_names):
      domain = multiwoz_utils.get_domain(slot_name)

      # Decide description for this slot.
      # slot_descriptions.json has multiple descriptions for each slot, for now
      # only use the first one.
      full_desc = slot_descriptions[slot_name][0]
      if options.description_type == 'full_desc':
        desc = f'{i}{options.delimiter}{full_desc}'
      elif options.description_type == 'full_desc_with_domain':
        desc = f'{i}{options.delimiter}{domain}-{full_desc}'
      elif options.description_type == 'item_name':
        desc = f'{i}{options.delimiter}{slot_name}'
      elif options.description_type == 'shuffled_item_name':
        # Make a copy of the slot name and shuffle it
        slot_name_shuffled = list(slot_name)
        random.shuffle(slot_name_shuffled)
        slot_name_shuffled = ''.join(slot_name_shuffled)
        desc = f'{i}{options.delimiter}{slot_name_shuffled}'

      letters = list(string.ascii_lowercase)
      possible_values_shuffled = []
      slot = schema_info.slots_by_domain[domain][slot_name]
      # Optionally append multiple choice prompt for this slot's description.
      if options.multiple_choice != 'none' and slot.is_categorical:
        possible_values_shuffled = slot.possible_values.copy()
        random.shuffle(possible_values_shuffled)
        assert len(possible_values_shuffled) < len(letters)

        if options.multiple_choice == 'a':
          desc_format_str = '{letter}) {value}'
        elif options.multiple_choice == '1a':
          desc_format_str = '{slot_id}{letter}) {value}'

        possible_values_pieces = []
        for letter, value in zip(letters, possible_values_shuffled):
          if options.description_type == 'shuffled_item_name':
            value_list = list(value)
            random.shuffle(value_list)
            value = ''.join(value_list)
          possible_values_pieces.append(
              desc_format_str.format(slot_id=i, letter=letter, value=value))

        desc += ' ' + ' '.join(possible_values_pieces)
      prefix_pieces.append(desc)

      # Generate target state string for this slot.
      if slot_name in belief_state:
        values = belief_state[slot_name]
        if '|' in values:
          values = values.split('|')
        elif '>' in values:
          values = values.split('>')
        elif '<' in values:
          values = values.split('<')
        elif options.multiwoz_version != '2.2':
          # In 2.2, multiple possible values are given. Consider a list of
          # values to accommodate.
          values = [values]

        # Convert this target value to categorical if required.
        if options.multiple_choice != 'none' and slot.is_categorical:
          values = [
              _multiple_choice_answer(i, letters, possible_values_shuffled, val)
              for val in values
          ]

        values_str = ' | '.join(values)
        state_pieces.append(f'{i}{options.delimiter}{values_str}')

    # Make sure all slots in the belief state end up in the target.
    if len(state_pieces) != len(belief_state):
      raise ValueError('Len of state_pieces must equal len of belief state.'
                       f'len(state_pieces): {len(state_pieces)}. '
                       f'len(belief_state): {len(belief_state)}.')

    prefix_str = ' '.join(prefix_pieces)
    state_str = '[states] ' + ' '.join(state_pieces)

    return TextToTextExample(
        src=f'{prefix_str} {history_str.strip()}'.strip(),
        # TODO(jeffreyzhao): Support intents, requested slots from MultiWOZ 2.2.
        # For now add empty "[intents] [req_slots]" to be consistent with SGD.
        tgt=f'{state_str.strip()} [intents] [req_slots]',
        dialog_id=dialog_id,
        turn=turn)

  examples = []
  for dialog_id, dialog in dialogs_by_id.items():
    history_str = ''

    for turn_num, turn in enumerate(dialog.turns):
      is_system = turn_num % 2 == 1
      speaker = 'system' if is_system else 'user'
      utterance = turn.utterance.strip().replace('\t', ' ')
      belief_state = turn.belief_state

      # State, action, response only appear at system turns.
      domains_in_turn = multiwoz_utils.extract_domains(belief_state)
      if is_system:
        if domains_in_turn & options.blocked_domains:
          continue
        examples.append(
            _process_one_turn(dialog_id, turn_num, belief_state, history_str,
                              domains_in_turn, slot_descriptions))
      history_str += f'[{speaker}] {utterance} '

  return examples


def main(_):
  random.seed(FLAGS.random_seed)
  multiwoz_data = multiwoz_utils.load_data(
      data_path=FLAGS.multiwoz_dir,
      multiwoz_version=FLAGS.multiwoz_version,
      is_trade=False)
  schema_info = multiwoz_utils.load_schema(FLAGS.schema_file)
  options = Options(
      multiwoz_version=FLAGS.multiwoz_version,
      description_type=FLAGS.description_type,
      delimiter=FLAGS.delimiter,
      multiple_choice=FLAGS.multiple_choice,
      use_active_domains_only=FLAGS.use_active_domains_only,
      blocked_domains=set(FLAGS.blocked_domains))

  split_to_examples = {
      'train':
          create_schemaless_data(multiwoz_data.train_json, schema_info,
                                 multiwoz_data.slot_descriptions, options),
      'dev':
          create_schemaless_data(multiwoz_data.dev_json, schema_info,
                                 multiwoz_data.slot_descriptions, options),
      'test':
          create_schemaless_data(multiwoz_data.test_json, schema_info,
                                 multiwoz_data.slot_descriptions, options)
  }
  split_to_examples['dev_test'] = (
      split_to_examples['dev'] + split_to_examples['test'])

  for split, examples in split_to_examples.items():
    text_to_text_utils.write_data(
        examples, os.path.join(FLAGS.output_dir, f'{split}.tfrecord'))


if __name__ == '__main__':
  flags.mark_flag_as_required('multiwoz_dir')
  flags.mark_flag_as_required('output_dir')
  flags.mark_flag_as_required('schema_file')
  tf.app.run(main)
