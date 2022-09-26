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

"""Utils for processing "Show Don't Tell" dialogue data."""

import collections
import random
import string
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from state_tracking.show_dont_tell import sdt_prompts
from state_tracking.utils import sgd_utils

Prompt = sdt_prompts.Prompt
Schemas = List[Any]
SourceToTarget = Dict[str, str]
SLOT_VALUE_DELIMITER = '='
_MCQ_OPTION_IDS = string.ascii_lowercase
_IS_CAT_VAL_IDENTIFIER = 'of possible values'


def generate_prompt_str(
    keys: Sequence[str],
    key_to_prompts: Optional[Mapping[str, Sequence[Prompt]]],
    prompt_indices: Optional[Sequence[str]] = None,
    mcq_cat_vals: bool = True,
    randomize_slots: bool = True,
    randomize_cat_vals: bool = True,
    use_slot_ids: bool = False,
    key_to_schema: Optional[Mapping[str, sgd_utils.Schema]] = None
) -> Tuple[str, List[str], Optional[Mapping[str, Mapping[str, str]]]]:
  """Generates the prompt string for an example.

  Args:
    keys: The keys for looking up relevant prompts. Usually a service/domain
    key_to_prompts: A map from key to a list of prompts
    prompt_indices: The desired indices of prompts to use for each service
    mcq_cat_vals: If true, enumerate categorical slot values in the form of a
      multiple choice question
    randomize_slots: If true, randomize the slot order
    randomize_cat_vals: If true, randomize the order of categorical values
    use_slot_ids: If true, replace slot names with numeric slot IDs
    key_to_schema: A map from key to schema. If present, use schema to add slot
      descriptions to prompt

  Returns:
    A formatted prompt string, a list of ordered slots, and a dict of slot name
    to MCQ ID of categorical value to categorical value.
  """

  def _convert_cat_val_prompt_to_mcq(
      value_str: str, cat_val_to_id: MutableMapping[str,
                                                    str]) -> Tuple[str, str]:
    """Convert categorical value string to MCQ format.

    Also updates in place the mapping of MCQ option IDs to values if needed.

    Args:
      value_str: Categorical value string in the form of '<val> of possible
        values <val1>, <val2> ...'
      cat_val_to_id: Dict of options for categorical value to MCQ ID. This is
        initialized/updated in this function when first called and reused
        thereafter.

    Returns:
      A multiple-choice letter representing a value and a string of all possible
      options. E.g. Tuple('b', 'a) True b) False')
    """
    value, options_str = [
        s.strip() for s in value_str.split(_IS_CAT_VAL_IDENTIFIER)
    ]
    options_list = options_str.split(', ')

    # Construct mapping of cat val options to IDs if needed.
    if not cat_val_to_id:
      if randomize_cat_vals:
        random.shuffle(options_list)
      for o_idx, option in enumerate(options_list):
        cat_val_to_id[option] = _MCQ_OPTION_IDS[o_idx]

    id_to_cat_val = {v_id: cat_val for cat_val, v_id in cat_val_to_id.items()}

    if value in cat_val_to_id:
      value = cat_val_to_id[value]
    elif value != 'dontcare':
      raise ValueError('Invalid categorical value string: %s' % value_str)

    return value, ' '.join([
        f'{v_id}) {id_to_cat_val[v_id]}'
        for v_id in sorted(id_to_cat_val.keys())
    ])

  if not key_to_prompts:
    return '', [''], None

  # Validate prompt_indices
  if prompt_indices and any([not i.isdigit() for i in prompt_indices]):
    raise ValueError('Please specify prompt_indices as a list of integers, or '
                     'as `None` to use all available prompts. '
                     f'prompt_indices: {prompt_indices}')

  # Fetch prompts and flatten into a single list
  selected_prompts = []
  for key in keys:
    for p_idx, prompt in enumerate(key_to_prompts[key]):
      if not prompt_indices or str(p_idx) in prompt_indices:
        selected_prompts.append(prompt)
  if randomize_slots:
    random.shuffle(selected_prompts)

  # Construct mapping from slot to description, for SDT+D3ST combination
  if key_to_schema:
    slot_to_desc = {}
    for key in keys:
      for slot_json in key_to_schema[key]['slots']:
        name = slot_json['name']
        desc = slot_json['description']
        if name in slot_to_desc:
          raise ValueError(
              'Duplicate slots found in single example. Unable to create '
              f'1-to-1 map of slots to descriptions. slot = {name} '
              f'existing desc = {slot_to_desc[name]} ; new desc = {desc}')
        slot_to_desc[name] = desc

  # Create one string for each prompt example
  prompt_substrs = []
  global_ordered_slots = []
  if mcq_cat_vals:
    slot_to_cat_val_to_id = collections.defaultdict(dict)
  else:
    slot_to_cat_val_to_id = None

  for prompt in selected_prompts:
    # Fix the slot order for each prompt
    ordered_slots = list(prompt.slots.keys())
    if randomize_slots:
      random.shuffle(ordered_slots)
    global_ordered_slots.extend(ordered_slots)

    # Construct D3ST prompt, if key_to_schema provided
    if key_to_schema:
      single_d3st_slot_strs = []
      for idx, slot in enumerate(ordered_slots):
        slot_identifier = idx if use_slot_ids else slot
        desc = slot_to_desc[slot]
        value_str = prompt.slots[slot]
        if mcq_cat_vals and _IS_CAT_VAL_IDENTIFIER in value_str:
          _, possible_mc_vals_str = _convert_cat_val_prompt_to_mcq(
              value_str, slot_to_cat_val_to_id[slot])
          desc = f'{desc} {possible_mc_vals_str}'

        single_d3st_slot_strs.append(
            f'{slot_identifier}{SLOT_VALUE_DELIMITER}{desc}')
      prompt_substrs.append(' '.join(single_d3st_slot_strs))

    # Construct SDT prompt, one slot string at a time
    single_sdt_slot_strs = []
    for idx, slot in enumerate(ordered_slots):
      slot_identifier = idx if use_slot_ids else slot
      value_str = prompt.slots[slot]
      if mcq_cat_vals and _IS_CAT_VAL_IDENTIFIER in value_str:
        mc_val_str, possible_mc_vals_str = _convert_cat_val_prompt_to_mcq(
            value_str, slot_to_cat_val_to_id[slot])
        value_str = f'{mc_val_str} {_IS_CAT_VAL_IDENTIFIER} {possible_mc_vals_str}'
      single_sdt_slot_strs.append(
          f'{slot_identifier}{SLOT_VALUE_DELIMITER}{value_str}')

    sdt_slot_str = ' '.join(single_sdt_slot_strs)
    prompt_substrs.append(f'[EXAMPLE] {prompt.utt} [slots] {sdt_slot_str}')

  prompt_str = ' '.join(prompt_substrs)

  return prompt_str, global_ordered_slots, slot_to_cat_val_to_id


def generate_context_str(history_utterances: List[str],
                         context_format: str) -> str:
  """Generates the context string for an example."""
  if context_format == 'dialogue':
    context_str = '[CONTEXT] ' + ' '.join(history_utterances)
  else:
    raise ValueError(f'Invalid context format specified: {context_format}')

  return context_str


def generate_target_str(dialogue_state: Dict[str, List[str]],
                        ordered_slots: List[str],
                        slot_to_cat_val_to_id: Mapping[str, Mapping[str, str]],
                        target_format: str,
                        use_slot_ids: bool = False) -> str:
  """Generates the target string for an example.

  Args:
    dialogue_state: A mapping of slot names to values
    ordered_slots: A list of ordered slots used to decide the target slot order
    slot_to_cat_val_to_id: Dict of slot names to categorical slot values to
      value IDs as in an MCQ
    target_format: The desired format for the generated target string
    use_slot_ids: If true, replace slot names with numeric slot IDs

  Returns:
    A formatted target string
  """
  # All slots are presented in the target string
  if target_format in ['all', 'active']:
    slot_strs = []
    for idx, slot in enumerate(ordered_slots):
      slot_value = dialogue_state.get(slot)  # List[str]

      # TODO(harrisonlee): Check if we should be using first slot value for SGD
      if slot_value:
        slot_value = slot_value[0]

      # Inactive slots - for 'all', value is 'none'; for 'active', skip it
      else:
        if target_format == 'all':
          slot_value = 'none'
        elif target_format == 'active':
          continue

      slot_identifier = idx if use_slot_ids else slot
      if (slot_to_cat_val_to_id and slot in slot_to_cat_val_to_id and
          slot_value in slot_to_cat_val_to_id[slot]):
        slot_value = slot_to_cat_val_to_id[slot][slot_value]

      slot_strs.append(f'{slot_identifier}{SLOT_VALUE_DELIMITER}{slot_value}')
    target_str = '[state] ' + ' '.join(slot_strs)
  elif not target_format:
    target_str = ''
  else:
    raise ValueError(f'Invalid target format specified: {target_format}')

  return target_str


def _create_schema_name_map(
    source_subdir_to_schemas: Dict[str, Schemas],
    target_subdir_to_schemas: Dict[str, Schemas]
) -> Tuple[SourceToTarget, Dict[str, SourceToTarget], Dict[str,
                                                           SourceToTarget]]:
  """Creates mapping from source to target schema element names.

  Args:
    source_subdir_to_schemas: A map from source data subdirectory its schemas
    target_subdir_to_schemas: A map from target data subdirectory its schemas

  Returns:
    A map from source to target service names, a map from service name to a
    sub-map from source to target slot names, and a map from service name to a
    sub-map from source to target intent names
  """
  service_to_name = {}
  service_slot_to_name = collections.defaultdict(dict)
  service_intent_to_name = collections.defaultdict(dict)

  # Note: this relies on schema order between source and target matching.
  for subdir, source_schemas in source_subdir_to_schemas.items():
    for idx, source_schema in enumerate(source_schemas):
      target_schema = target_subdir_to_schemas[subdir][idx]
      service = source_schema['service_name']

      # Skip processing if schema has already been processed.
      if service in service_to_name:
        continue

      # Service name.
      service_to_name[service] = target_schema['service_name']

      # Slot names.
      for s_idx, slot in enumerate(source_schema['slots']):
        service_slot_to_name[service][
            slot['name']] = target_schema['slots'][s_idx]['name']

      # Intent names.
      for i_idx, intent in enumerate(source_schema['intents']):
        service_intent_to_name[service][
            intent['name']] = target_schema['intents'][i_idx]['name']

  return service_to_name, service_slot_to_name, service_intent_to_name


def create_sgdx_prompts(service_to_prompts: Mapping[str, List[Prompt]],
                        sgd_dir: str, sgdx_dir: str) -> Dict[str, List[Prompt]]:
  """Creates a SGD-X version of SGD prompts.

  Note that currently does not support intents.
  Args:
    service_to_prompts: A map from SGD services to lists of prompts
    sgd_dir: Path to SGD dataset directory
    sgdx_dir: Path to a SGD-X variant dataset directory

  Returns:
    A map from SGD services to SGD-X versions of prompts
  """

  # Load original schemas
  orig_subdir_to_schemas = {}
  for subdir in ['train', 'dev', 'test']:
    sgd_utils.load_schemas_to_dict(sgd_dir, subdir, orig_subdir_to_schemas)

  # Load variant schemas
  var_subdir_to_schemas = {}
  for subdir in ['train', 'dev', 'test']:
    sgd_utils.load_schemas_to_dict(sgdx_dir, subdir, var_subdir_to_schemas)

  # Create mappings from original to variant schema element names.
  service_to_name, service_slot_to_name, _ = _create_schema_name_map(
      orig_subdir_to_schemas, var_subdir_to_schemas)

  # Create new service_to_prompts map with SGD-X elements
  sgdx_service_to_prompts = {}
  for service, prompts in service_to_prompts.items():
    sgdx_prompts = []
    for prompt in prompts:
      sgdx_slots = collections.OrderedDict()
      for slot, value in prompt.slots.items():
        sgdx_slot_name = service_slot_to_name[service][slot]
        sgdx_slots[sgdx_slot_name] = value
      sgdx_prompts.append(Prompt(utt=prompt.utt, slots=sgdx_slots))
    sgdx_service = service_to_name[service]
    sgdx_service_to_prompts[sgdx_service] = sgdx_prompts

  return sgdx_service_to_prompts
