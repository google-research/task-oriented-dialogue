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

r"""Create Show Don't Tell data from SGD Dataset.

Format: [example] <example dialogue> [slots] <slot names and values> \
[context] <current dialogue> -> [state] <dialogue state>

Example: [example] [user] can you find me a bus to lax? ... \
[slots] to_location=lax ... [context] [user] i'm looking for a bus to nyc. ... \
-> [state] to_location=nyc ...
"""

import collections
import dataclasses
import itertools
import os
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence

from absl import app
from absl import flags
from absl import logging
from state_tracking.show_dont_tell import sdt_prompts
from state_tracking.show_dont_tell import sdt_utils
from state_tracking.utils import sgd_utils
import tensorflow as tf

_INPUT_DIR = flags.DEFINE_string('input_dir', None,
                                 'Path to SGD data directory.')
_OUTPUT_PATH = flags.DEFINE_string('output_path', None, 'Path for output file.')
_SGDX_DIR = flags.DEFINE_string(
    'sgdx_dir', None, 'If set, create dialogue examples using SGD-X variants '
    'from this path. e.g. /path/to/sgdx/v1/')
_SUBDIRS = flags.DEFINE_list(
    'subdirs', 'train,dev,test',
    'Comma-separated list of dataset subdirectories to process.')
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
    'target_format', 'all', ['all', 'active'],
    'Format of the target. "all" and '
    '"active" respectively refer to all and only active slots being present in '
    'the target.')
_LOWERCASE = flags.DEFINE_bool('lowercase', True,
                               'Whether to lowercase the generated example.')
_MCQ_CAT_VALS = flags.DEFINE_bool(
    'mcq_cat_vals', False,
    'Whether to enumerate categorical values in the form of a multiple choice '
    'question in the prompt string.')
_RANDOMIZE_SLOTS = flags.DEFINE_bool(
    'randomize_slots', True, 'Whether to randomize slot order of the prompt.')
_RANDOMIZE_CAT_VALS = flags.DEFINE_bool(
    'randomize_cat_vals', True,
    'Whether to randomize order of categorical values in prompt.')
_USE_SLOT_IDS = flags.DEFINE_bool(
    'use_slot_ids', False, 'Whether to use '
    'numeric slot IDs in place of slot names in '
    'the input and output strings.')
_DATA_PERCENT = flags.DEFINE_float(
    'data_percent', 0.0, 'If not 0.0, only write this proportion of data, and '
    'discard the rest of the examples. For data efficiency experiments. '
    'Not compatible with k_shot.')
_K_SHOT = flags.DEFINE_integer(
    'k_shot', 0, 'If not 0, sample this many examples from each service. '
    'For data efficiency experiments. Not compatible with data_percent.')
_USE_SLOT_DESCS = flags.DEFINE_bool(
    'use_slot_descs', False, 'Whether to add D3ST descriptions to prompt.')

Prompt = sdt_prompts.Prompt

Schemas = sgd_utils.Schemas
DialoguesDict = sgd_utils.DialoguesDict
RAND_SEED = 123
USER_SPEAKER = 'USER'
SYSTEM_SPEAKER = 'SYSTEM'
USER_TOK = '[user]'
SYS_TOK = '[system]'
SLOT_VALUE_DELIMITER = '='
INPUT_TARGET_SEP = '\t'

_PROMPTS_MAP = {
    'separated': sdt_prompts.SGD_SEPARATED_ANNOTATION_PROMPTS,
}


@dataclasses.dataclass
class Options:
  """A dataclass to store configurations for data generation."""
  sgd_dir: str
  sgdx_dir: Optional[str]
  prompt_format: Optional[str]
  context_format: str
  target_format: str
  lowercase: bool
  mcq_cat_vals: bool
  randomize_slots: bool
  randomize_cat_vals: bool
  use_slot_ids: bool
  prompt_indices: List[str]


@dataclasses.dataclass
class Example:
  """Dataclass for single SDT example.

  Attributes:
    example_str: The example string.
    services: The services this example belongs to.
  """
  example_str: str
  services: List[str]


def _generate_utt_str(utterance: str, speaker: str) -> str:
  """Generates the utterance string for an example."""
  if speaker == USER_SPEAKER:
    prefix = USER_TOK
  elif speaker == SYSTEM_SPEAKER:
    prefix = SYS_TOK
  else:
    raise ValueError(f'Speaker must be one of {USER_SPEAKER} '
                     f'or {SYSTEM_SPEAKER}. Found {speaker}')

  # Occasionally some examples include newlines in the middle
  utterance = utterance.replace('\n', ' ')

  return ' '.join([prefix, utterance])


def build_example(input_strs: Sequence[str], target_str: str,
                  additional_strs: Sequence[str], services: Sequence[str],
                  lowercase: bool) -> Example:
  """Builds a single example in TSV format."""
  example_str = ' '.join(input_strs) + INPUT_TARGET_SEP + target_str
  if additional_strs:
    example_str += INPUT_TARGET_SEP + INPUT_TARGET_SEP.join(additional_strs)

  if lowercase:
    example_str = example_str.lower()

  return Example(example_str=example_str.strip(), services=list(services))


def create_examples_from_dialogue(dialogue: Mapping[
    str, Any], service_to_prompts: Optional[Dict[str, List[Prompt]]],
                                  service_to_schema: Mapping[str,
                                                             sgd_utils.Schema],
                                  options: Options) -> List[Example]:
  """Returns example strings created from a dialogue.

  Args:
    dialogue: A single dialogue containing multiple turns and frames
    service_to_prompts: A map from SGD service to a list of prompts
    service_to_schema: A map from SGD service to schema
    options: An object containing various options related to example generation
  """
  utt_strs = []
  example_strs = []

  for turn_idx, turn in enumerate(dialogue['turns']):

    # Format utterances
    utt_strs.append(
        _generate_utt_str(utterance=turn['utterance'], speaker=turn['speaker']))

    # Don't create examples out of system turns for DST
    if turn['speaker'] != USER_SPEAKER:
      continue

    for frame_idx, frame in enumerate(turn['frames']):

      # Create prompt
      prompt_str, ordered_slots, slot_to_cat_val_to_id = sdt_utils.generate_prompt_str(
          keys=[frame['service']],
          key_to_prompts=service_to_prompts,
          prompt_indices=options.prompt_indices,
          mcq_cat_vals=options.mcq_cat_vals,
          randomize_slots=options.randomize_slots,
          randomize_cat_vals=options.randomize_cat_vals,
          use_slot_ids=options.use_slot_ids,
          key_to_schema=service_to_schema)

      # Create context
      context_str = sdt_utils.generate_context_str(utt_strs,
                                                   options.context_format)

      # Create target
      target_str = sdt_utils.generate_target_str(
          dialogue_state=frame['state']['slot_values'],
          ordered_slots=ordered_slots,
          slot_to_cat_val_to_id=slot_to_cat_val_to_id,
          target_format=options.target_format,
          use_slot_ids=options.use_slot_ids)

      example_strs.append(
          build_example(
              input_strs=[prompt_str, context_str],
              target_str=target_str,
              additional_strs=[
                  dialogue['dialogue_id'],
                  str(turn_idx),
                  str(frame_idx)
              ],
              services=dialogue['services'],
              lowercase=options.lowercase))

  return example_strs


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if _DATA_PERCENT.value > 0.0 and _K_SHOT.value > 0:
    raise ValueError('Only one of data_percent and k_shot can be specified!')

  # Set random seed
  random.seed(RAND_SEED)

  options = Options(
      sgd_dir=_INPUT_DIR.value,
      sgdx_dir=_SGDX_DIR.value,
      prompt_format=_PROMPT_FORMAT.value,
      context_format=_CONTEXT_FORMAT.value,
      target_format=_TARGET_FORMAT.value,
      mcq_cat_vals=_MCQ_CAT_VALS.value,
      randomize_slots=_RANDOMIZE_SLOTS.value,
      randomize_cat_vals=_RANDOMIZE_CAT_VALS.value,
      lowercase=_LOWERCASE.value,
      use_slot_ids=_USE_SLOT_IDS.value,
      prompt_indices=_PROMPT_INDICES.value)

  # Load dataset - SGD-X if provided, otherwise SGD
  sgd_data_dir = options.sgdx_dir or options.sgd_dir
  subdir_to_schema, subdir_to_dialogues = sgd_utils.load_dataset(
      data_dir=sgd_data_dir, subdirs=_SUBDIRS.value)

  # If enabled, create map from service to schema for adding D3ST descriptions
  if _USE_SLOT_DESCS.value:
    service_to_schema = sgd_utils.dedupe_and_unnest_schemas(subdir_to_schema)
  else:
    service_to_schema = False

  # Fetch prompts and replace with SGD-X version if applicable
  if options.prompt_format:
    service_to_prompts = _PROMPTS_MAP[options.prompt_format]
    if options.sgdx_dir:
      service_to_prompts = sdt_utils.create_sgdx_prompts(
          service_to_prompts, options.sgd_dir, options.sgdx_dir)
  else:
    service_to_prompts = None

  # Create output directory if needed
  if not tf.io.gfile.isdir(os.path.dirname(_OUTPUT_PATH.value)):
    tf.io.gfile.makedirs(os.path.dirname(_OUTPUT_PATH.value))

  # Loop through dialogues and create examples
  with tf.io.gfile.GFile(_OUTPUT_PATH.value, 'w') as outfile:
    examples = []
    for subdir, dfile_to_dialogues in subdir_to_dialogues.items():
      logging.info('Processing subdir %s', subdir)

      for dfile, dialogues in dfile_to_dialogues.items():
        logging.info('Processing file %s', dfile)

        for dialogue in dialogues:
          examples.extend(
              create_examples_from_dialogue(dialogue, service_to_prompts,
                                            service_to_schema, options))

    # Optionally sample a proportion of examples only
    if _DATA_PERCENT.value > 0.0:
      examples = random.sample(examples,
                               int(_DATA_PERCENT.value * len(examples)))
    elif _K_SHOT.value > 0:
      # A dict of service to a list of examples belonging to that service.
      service_to_examples = collections.defaultdict(list)
      for example in examples:
        for service in example.services:
          service_to_examples[service].append(example)

      # Sample K examples from each service
      examples = itertools.chain(*[
          random.sample(examples_by_service, k=_K_SHOT.value)
          for examples_by_service in service_to_examples.values()
      ])

    # Write example strings to file
    for e in examples:
      outfile.write(f'{e.example_str}\n')


if __name__ == '__main__':
  flags.mark_flag_as_required('input_dir')
  flags.mark_flag_as_required('output_path')
  app.run(main)
