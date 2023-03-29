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

r"""Create Show Don't Tell data from Multiwoz Dataset.

Format: [example] <example dialogue> [slots] <slot names and values> \
[context] <current dialogue> -> [state] <dialogue state>
e.g. `[example] [user] can you find me a train to lax? ... \
[slots] train-destination=lax ... [context] [user] i'm looking for a train
to nyc. ... \ -> [state] train-destination=nyc ...`
"""

import collections
import dataclasses
import os
import random
from typing import Dict, List, Set

from absl import app
from absl import flags
from state_tracking.show_dont_tell import sdt_prompts
from state_tracking.show_dont_tell import sdt_utils
from state_tracking.utils import multiwoz_utils
from state_tracking.utils import text_to_text_utils

FLAGS = flags.FLAGS

_INPUT_DIR = flags.DEFINE_string('input_dir', None,
                                 'Path to the original MultiWOZ datasets.')
_OUTPUT_DIR = flags.DEFINE_string('output_dir', None, 'Output file path.')
_SCHEMA_FILE = flags.DEFINE_string('schema_file', None,
                                   'MultiWOZ schema file in 2.2/SGD format.')
_MULTIWOZ_VERSION = flags.DEFINE_enum('multiwoz_version', '2.1',
                                      ('2.1', '2.2', '2.3', '2.4'),
                                      'MultiWOZ dataset version.')
_IS_TRADE = flags.DEFINE_bool('is_trade', True,
                              'Whether the data is TRADE-preprocessed or not.')
_PROMPT_FORMAT = flags.DEFINE_enum(
    'prompt_format', 'separated', ['separated'],
    'Format of the prompt for priming. '
    '"separated" means a dialogue followed by a separate string of slots.')
_PROMPT_INDICES = flags.DEFINE_list(
    'prompt_indices', None,
    'Indices of the prompts for each service to be used for generating '
    'examples. Specify one or more numeric indices (starting from 0), or `None` '
    'to use all prompts for a given service.')
_CONTEXT_FORMAT = flags.DEFINE_enum('context_format', 'dialogue', ['dialogue'],
                                    'Format of the dialogue context.')
_TARGET_FORMAT = flags.DEFINE_enum(
    'target_format', 'all', ['all'], 'Format of the target. "all" refers to'
    'all slots being in the target.')
_LOWERCASE = flags.DEFINE_bool('lowercase', True,
                               'Whether to lowercase the generated example.')
_MCQ_CAT_VALS = flags.DEFINE_bool(
    'mcq_cat_vals', False,
    'Whether to enumerate categorical values in the form of a multiple choice '
    'question in the prompt.')
_RANDOMIZE_SLOTS = flags.DEFINE_bool(
    'randomize_slots', True, 'Whether to randomize slot order of the prompt.')
_RANDOMIZE_CAT_VALS = flags.DEFINE_bool(
    'randomize_cat_vals', True,
    'Whether to randomize order of categorical values in prompt.')
_SHUFFLE = flags.DEFINE_bool(
    'shuffle', True, 'Whether to randomly shuffle examples before writing out.')
_USE_ACTIVE_DOMAINS_ONLY = flags.DEFINE_bool(
    'use_active_domains_only', False,
    'If true, only include domains that are active in this dialogue in prompt.')
_BLOCKED_DOMAINS = flags.DEFINE_list(
    'blocked_domains', [], 'Don\'t include these domains '
    'if set. This is used to run zero-shot '
    'cross-domain experiments as in paper '
    'https://aclanthology.org/2021.naacl-main.448.pdf.')


# Use OrderedDict for JSON to preserve field order.
Json = collections.OrderedDict
MultiwozData = multiwoz_utils.MultiwozData
SchemaInfo = multiwoz_utils.SchemaInfo
TextToTextExample = text_to_text_utils.TextToTextExample
MULTIWOZ_DOMAINS = [
    'attraction',
    'bus',
    'hospital',
    'hotel',
    'restaurant',
    'taxi',
    'train',
]
USER_TOK = '[user]'
SYS_TOK = '[system]'

_PROMPTS_MAP = {
    'separated': sdt_prompts.MW_SEPARATED_ANNOTATION_PROMPTS,
}


@dataclasses.dataclass
class Options:
  """Options for generating SDT examples."""
  multiwoz_version: str
  is_trade: bool
  prompt_format: str
  prompt_indices: List[str]
  context_format: str
  target_format: str
  mcq_cat_vals: bool
  randomize_slots: bool
  randomize_cat_vals: bool
  use_active_domains_only: bool
  blocked_domains: Set[str]
  lowercase: bool


def _normalize_multiwoz_slot_values(
    dialogue_state: Dict[str,
                         str], multiwoz_version: str) -> Dict[str, List[str]]:
  """Normalizes multiwoz slot values into a common format."""
  new_state = {}

  for slot_name, values in dialogue_state.items():
    if '|' in values:
      values = values.split('|')
    elif '>' in values:
      values = values.split('>')
    elif '<' in values:
      values = values.split('<')
    elif multiwoz_version != '2.2':
      # Put values into a list to accommodate 2.2 format giving multiple values
      values = [values]

    if not isinstance(values, list):
      raise ValueError('"values" for a slot must be of list type. Actual: '
                       f'{type(values)}. values: {values}')

    new_state[slot_name] = values

  return new_state


def _process_one_turn(dialog_id: str, turn: int, belief_state: Dict[str, str],
                      history_utterances: List[str],
                      options: Options) -> TextToTextExample:
  """Processes a single dialogue turn into a `TextToTextExample`."""
  # Fetch prompts
  domain_to_prompts = _PROMPTS_MAP[
      options.prompt_format] if options.prompt_format else None

  # Create prompt
  if options.use_active_domains_only:
    domains = list(multiwoz_utils.extract_domains(belief_state))
  else:
    domains = MULTIWOZ_DOMAINS
  prompt_str, ordered_slots, ordered_slot_to_cat_val_to_id, intent_to_id = sdt_utils.generate_prompt_str(
      keys=sorted(domains),
      key_to_prompts=domain_to_prompts,
      prompt_indices=options.prompt_indices,
      add_intents=False,
      mcq_cat_vals=options.mcq_cat_vals,
      mcq_intents=False,
      randomize_slots=options.randomize_slots,
      randomize_intents=False,
      randomize_cat_vals=options.randomize_cat_vals)

  # Create context
  context_str = sdt_utils.generate_context_str(history_utterances,
                                               options.context_format)

  # Create target
  norm_dialogue_state = _normalize_multiwoz_slot_values(
      belief_state, options.multiwoz_version)
  # MultiWoZ2.1 does not have active intents, hence setting to empty string.
  target_str = sdt_utils.generate_target_str(
      dialogue_state=norm_dialogue_state,
      active_intent='',
      add_intents=False,
      ordered_slots=ordered_slots,
      slot_to_cat_val_to_id=ordered_slot_to_cat_val_to_id,
      intent_to_id=intent_to_id,
      target_format=options.target_format,
      use_slot_ids=False)

  # Lowercase
  if options.lowercase:
    prompt_str = prompt_str.lower()
    context_str = context_str.lower()
    target_str = target_str.lower()

  return TextToTextExample(
      src=' '.join([prompt_str, context_str.strip()]).strip(),
      tgt=target_str,
      dialog_id=dialog_id,
      turn=turn)


def create_sdt_examples(json_data: Json,
                        options: Options) -> List[TextToTextExample]:
  """Converts raw MultiWOZ data into "Show Don't Tell" examples."""
  examples = []

  for dialog_id, dialog_json in json_data.items():
    history_utterances = []

    dialog_key = 'dialogue' if options.is_trade else 'log'
    belief_state_key = 'belief_state' if options.is_trade else 'metadata'
    for turn, utterance_json in enumerate(dialog_json[dialog_key]):
      # Process utterance json
      if options.is_trade:
        sys_utt = utterance_json['system_transcript'].strip().replace('\t', ' ')
        user_utt = utterance_json['transcript'].strip().replace('\t', ' ')
        if turn == 0:
          history_utterances.append(f'[user] {user_utt}')
        else:
          history_utterances.append(f'[system] {sys_utt} [user] {user_utt}')
        is_system = True
      else:
        is_system = turn % 2 == 1

      belief_state = multiwoz_utils.extract_belief_state(
          metadata_json=utterance_json[belief_state_key],
          is_trade=options.is_trade)

      # State, action, and response only in system turns for non-TRADE data
      if is_system:
        # Skip turns if a blocked domain appears (including adding to history)
        domains_in_turn = multiwoz_utils.extract_domains(belief_state)
        if options.blocked_domains & domains_in_turn:
          continue
        examples.append(
            _process_one_turn(dialog_id, turn, belief_state, history_utterances,
                              options))

      # Update history_utterances for non-TRADE data
      if not options.is_trade:
        utterance = utterance_json['text'].strip().replace('\t', ' ').replace(
            '\n', ' ')
        history_utterances.append(f'{SYS_TOK if is_system else USER_TOK} '
                                  f'{utterance}')

  return examples


def main(_):
  multiwoz_data = multiwoz_utils.load_data(
      data_path=_INPUT_DIR.value,
      multiwoz_version=_MULTIWOZ_VERSION.value,
      is_trade=_IS_TRADE.value)

  options = Options(
      multiwoz_version=_MULTIWOZ_VERSION.value,
      is_trade=_IS_TRADE.value,
      prompt_format=_PROMPT_FORMAT.value,
      prompt_indices=_PROMPT_INDICES.value,
      context_format=_CONTEXT_FORMAT.value,
      target_format=_TARGET_FORMAT.value,
      mcq_cat_vals=_MCQ_CAT_VALS.value,
      randomize_slots=_RANDOMIZE_SLOTS.value,
      randomize_cat_vals=_RANDOMIZE_CAT_VALS.value,
      use_active_domains_only=_USE_ACTIVE_DOMAINS_ONLY.value,
      blocked_domains=set(_BLOCKED_DOMAINS.value),
      lowercase=_LOWERCASE.value)

  # Create SDT examples
  split_to_examples = {
      'train': create_sdt_examples(multiwoz_data.train_json, options),
      'dev': create_sdt_examples(multiwoz_data.dev_json, options),
      'test': create_sdt_examples(multiwoz_data.test_json, options),
  }

  # Write out examples
  if _SHUFFLE.value:
    for examples in split_to_examples.values():
      random.shuffle(examples)
  split_to_examples['dev_test'] = (
      split_to_examples['dev'] + split_to_examples['test'])
  for split, examples in split_to_examples.items():
    text_to_text_utils.write_data(
        examples, os.path.join(_OUTPUT_DIR.value, f'{split}.tfrecord'))


if __name__ == '__main__':
  flags.mark_flag_as_required('input_dir')
  flags.mark_flag_as_required('output_dir')
  flags.mark_flag_as_required('schema_file')
  app.run(main)
