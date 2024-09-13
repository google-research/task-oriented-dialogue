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

"""Create SGD AnyTOD data."""

import collections
import copy
import dataclasses
import enum
import json
import os
import random
import string
import time
from typing import Generic, Iterator, Optional, TypeVar

from absl import app
from absl import flags
import ordered_set
import tensorflow as tf


class Mode(enum.Enum):
  ZOMBIE = 'zombie'
  ZOMBIE_HISTORY = 'zombie_history'
  ZOMBIE_HISTORY_2PASS = 'zombie_history_2pass'
  ZOMBIE_HISTORY_2PASS_2NDONLY = 'zombie_history_2pass_2ndonly'
  ZOMBIE_HISTORY_2PASS_EVAL = 'zombie_history_2pass_eval'


_INPUT_DIR = flags.DEFINE_string('input_dir', '', 'Input directory.')
_OUTPUT_DIR = flags.DEFINE_string('output_dir', '', 'Output directory.')
_MODE = flags.DEFINE_enum_class(
    'mode', Mode.ZOMBIE_HISTORY_2PASS, Mode, 'Mode.'
)
_CAT_SLOTS = flags.DEFINE_bool('cat_slots', True, 'Support categorical slots.')
_SHUFFLE = flags.DEFINE_bool(
    'shuffle', True, 'Should we shuffle training examples and indices?'
)
_FIX_TAGS = flags.DEFINE_bool('fix_tags', False, 'If true, use fixed tags.')

Slot = str
Json = dict
Action = tuple[str, Optional[str]]

T = TypeVar('T')


class Ordering(Generic[T]):
  """Provides an ordering for a list of entities."""

  def __init__(self, entities: list[T], shuffle_idxs: bool = True):
    self._idx_to_entity = copy.deepcopy(entities)
    if shuffle_idxs:
      random.shuffle(self._idx_to_entity)
    self._entity_to_idx = {e: i for i, e in enumerate(self._idx_to_entity)}

  def __contains__(self, e: T) -> bool:
    return e in self._idx_to_entity

  def get_idx(self, e: T) -> int:
    return self._entity_to_idx[e]

  def get_entity(self, idx: int) -> T:
    return self._idx_to_entity[idx]

  def __iter__(self) -> Iterator[tuple[int, T]]:
    return enumerate(self._idx_to_entity)

  def tolist(self) -> list[T]:
    return self._idx_to_entity


def _read_file(filename: str) -> str:
  with tf.io.gfile.GFile(filename) as f:
    return f.read()


def load_sgd_data(data_dir: str) -> tuple[list[Json], Json]:
  """Load the JSONs for a SGD split."""
  dialogs = []
  filenames = tf.io.gfile.glob(os.path.join(data_dir, 'dialogues_*.json'))

  start_time = time.time()
  # TODO(jeffreyzhao): Parallelize below.
  json_strs = [
      _read_file(f) for f in filenames
  ]
  print(
      f'Reading dialogues_*.json files took {time.time() - start_time} seconds.'
  )

  for js in json_strs:
    js = json.loads(js, object_hook=Json)
    for d in js:
      dialogs.append(d)
  with tf.io.gfile.GFile(os.path.join(data_dir, 'schema.json')) as f:
    schema = json.load(f, object_hook=Json)
  return dialogs, schema


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


@dataclasses.dataclass
class Example:
  """Example dataclass."""

  src: str
  tgt: str
  dialog_id: int
  turn: int
  frame: int
  service: str
  metadata: Json
  policy_table: list[Json]

  def build_tf_example(self) -> tf.train.Example:
    """Converts this Example dataclass into a tf.Example."""
    policy_table_json_str = json.dumps(self.policy_table)

    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'input': _bytes_feature(self.src.encode('utf-8')),
                'value': _bytes_feature(self.tgt.encode('utf-8')),
                'dialog_id': _bytes_feature(
                    str(self.dialog_id).encode('utf-8')
                ),
                'turn': _int64_feature(self.turn),
                'frame': _int64_feature(self.frame),
                'service': _bytes_feature(self.service.encode('utf-8')),
                'metadata': _bytes_feature(
                    json.dumps(self.metadata).encode('utf-8')
                ),
                'policy_table': _bytes_feature(
                    policy_table_json_str.encode('utf-8')
                ),
            }
        )
    )


def anytod_policy_function(belief_state, cur_intent, act_hist, api):
  """Define generic policy function for SGD."""
  if cur_intent == 'NONE':
    return set([('GOODBYE', None), ('REQ_MORE', None)])

  # This should not be expected for SGD since the first turn is always a user
  # turn, more of a safety check.
  if not act_hist:
    return set()

  last_turn_user_acts = act_hist[-1]

  # Get the intent signature to obtain required/optional slots etc.
  intent_signature = None
  for intent_sig in api['intents']:
    if intent_sig['name'].lower() == cur_intent.lower():
      intent_signature = intent_sig
  if not intent_signature:
    raise ValueError(f'Intent {cur_intent} not found in schema.')

  required_slots = set(intent_signature['required_slots'])
  optional_slots = set(intent_signature['optional_slots'])
  offered_slots = set(intent_signature['offered_slots'])
  is_transactional = intent_signature['is_transactional']

  # Identify filled and unfilled slots.
  filled_slots = set(belief_state.keys())
  last_turn_informed_slots = set()
  for user_action in last_turn_user_acts:
    if user_action[0] == 'INFORM':
      last_turn_informed_slots.add(user_action[1])
      filled_slots.add(user_action[1])
  all_required_slots_filled = filled_slots >= required_slots
  unfilled_required_slots = required_slots - filled_slots

  # Set of recommended system actions to be returned.
  sys_act_recs = set()

  # Special case somewhat specific to the SGD dataset.
  num_informs = len(last_turn_informed_slots)
  has_negate = bool([act for act in last_turn_user_acts if act[0] == 'NEGATE'])
  # If a user negates a system confirmation for a transactional API and
  # informs exactly two slots, only those two slots are confirmed again.
  if (
      is_transactional
      and has_negate
      and all_required_slots_filled
      and num_informs == 2
  ):
    sys_act_recs.update(
        [('CONFIRM', slot) for slot in last_turn_informed_slots]
    )
    return sys_act_recs

  # Else, general graph lookup.
  for user_action in last_turn_user_acts:
    act = user_action[0]
    # If user is informing preferences and an API call can be made.
    if (
        act in ['INFORM', 'INFORM_INTENT', 'AFFIRM_INTENT']
        and all_required_slots_filled
    ):
      # For transactional intents, the system confirms all slots, required and
      # optional.
      if is_transactional:
        sys_act_recs.update(
            [('CONFIRM', slot) for slot in required_slots.union(optional_slots)]
        )
      # For non-transactional intents, the system only offers offered slots
      # that uniquely identify an entity.
      else:
        sys_act_recs.update([('OFFER', slot) for slot in offered_slots])
        sys_act_recs.add(('INFORM_COUNT', None))
    # There are more required slots to request.
    elif act in ['INFORM', 'INFORM_INTENT', 'AFFIRM_INTENT']:
      sys_act_recs.update(
          [('REQUEST', slot) for slot in unfilled_required_slots]
      )
    # AFFIRM in a transactional API triggers an API call which may fail or
    # succeed: currently, actions corresponding to both are recommended.
    elif act == 'AFFIRM':
      if is_transactional and all_required_slots_filled:
        sys_act_recs.update([
            ('NOTIFY_SUCCESS', None),
            ('NOTIFY_FAILURE', None),
            ('REQ_MORE', None),
        ])
        sys_act_recs.update([('CONFIRM', slot) for slot in offered_slots])
      # Other than that, AFFIRM behaves similar to INFORM behavior as above.
      elif all_required_slots_filled:
        sys_act_recs.update([('OFFER', slot) for slot in offered_slots])
        sys_act_recs.add(('INFORM_COUNT', None))
      else:
        sys_act_recs.update(
            [('REQUEST', slot) for slot in unfilled_required_slots]
        )
    # User requesting a slot value.
    elif act == 'REQUEST':
      sys_act_recs.add(('INFORM', user_action[1]))
    # User requesting alternate options. If alternate options exist, the
    # system OFFERs them, else the call fails. Currently both sets of actions
    # are recommended.
    elif act == 'REQUEST_ALTS':
      sys_act_recs.update([('OFFER', slot) for slot in offered_slots])
      sys_act_recs.update([('NOTIFY_FAILURE', None), ('REQ_MORE', None)])
      if not is_transactional:
        sys_act_recs.add(('INFORM_COUNT', None))
    # If the user negates the system confirmation (in transactional APIs).
    elif act == 'NEGATE':
      sys_act_recs.add(('REQ_MORE', None))
    # User says bye.
    elif act == 'GOODBYE':
      sys_act_recs.add(('GOODBYE', None))
    # User is done, system asks if they need something else.
    elif act == 'THANK_YOU':
      sys_act_recs.add(('REQ_MORE', None))
    # SELECT may mean recommending a booking intent, or just the system asking
    # if the user needs something else.
    elif act == 'SELECT' and not is_transactional:
      sys_act_recs.add(('REQ_MORE', None))
      for intent in api['intents']:
        if intent['is_transactional']:
          sys_act_recs.add(('OFFER_INTENT', intent['name']))

  return sys_act_recs


class Converter:
  """Converter for SGD dialogs into examples."""

  def __init__(
      self,
      dialog_jsons: list[Json],
      schema: Json,
      mode: Mode,
      shuffle: bool,
      use_cat_slots: bool,
      fix_tags: bool,
  ):
    self._dialog_jsons = dialog_jsons
    self._schema = schema
    self._mode = mode
    self._shuffle = shuffle
    self._use_cat_slots = use_cat_slots
    self._fix_tags = fix_tags
    self._total_turns = 0
    self._total_turns_correct = 0

    self._services = collections.defaultdict(dict)
    self._slot_descs = collections.defaultdict(dict)
    self._slot_cat_vals = collections.defaultdict(dict)
    self._intent_descs = collections.defaultdict(dict)
    for service_json in schema:
      service = service_json['service_name']
      self._services[service] = service_json
      for slot_json in service_json['slots']:
        self._slot_descs[service][slot_json['name']] = slot_json['description']
        self._slot_cat_vals[service][slot_json['name']] = slot_json[
            'possible_values'
        ]
      for intent_json in service_json['intents']:
        self._intent_descs[service][intent_json['name']] = intent_json[
            'description'
        ]

    self._build_acts()
    # TODO(jeffreyzhao): Inferring policy graph currently unused.
    # self._build_graph()

  # # TODO(jeffreyzhao): AnyTOD natural language format.
  # def convert_to_anytodnl_format(dialog_json: Json, split: str, schema: Json):
  #   service_jsons = {}
  #   for service_json in schema:
  #     service_jsons[service_json['service_name']] = service_json

  #   examples = []
  #   convo_hist = []
  #   for turn, turn_json in enumerate(dialog_json['turns']):
  #     speaker, utt = turn_json['speaker'], turn_json['utterance']
  #     convo_hist.append((speaker, utt))
  #     convo_str = '\n'.join(
  #         f'{speaker}: {utt}' for speaker, utt in convo_hist)

  #     if speaker == 'USER':
  #       for frame, frame_json in enumerate(turn_json['frames']):
  #         service = frame_json['service']
  #         all_slots = []
  #         service_json = service_jsons[service]
  #         slot_descs = {}
  #         for slot_json in service_json['slots']:
  #           all_slots.append(slot_json['name'])
  #           slot_descs[slot_json['name']] = slot_json['description']

  #         bs = {}
  #         for slot, val in frame_json['state']['slot_values'].items():
  #           bs[slot] = val[0]
  #         slot_ord = Ordering(all_slots)
  #         bs_str = ('Question 1) Answer the following known params in the '
  #                   'conversation, or put down "none" if not known.\n')
  #         bs_tgt_str = []
  #         for _, slot in slot_ord:
  #           slot_desc = slot_descs[slot]
  #           val = bs.get(slot, 'none')
  #           bs_str += f' - {slot}: {slot_desc}\n'
  #           bs_tgt_str.append(f'{slot}={val}')
  #         bs_tgt_str = '1) ' + ', '.join(bs_tgt_str) + '\n'

  #         inp = f'{convo_str}\n{bs_str}\n'
  #         val = bs_tgt_str
  #         # print(inp)
  #         # print(val)
  #         # print('==========\n')
  #         metadata = {'slot_ord': slot_ord._idx_to_entity}
  #         examples.append(
  #             Example(inp, val, dialog_json['dialogue_id'], turn, frame,
  #                     service, metadata, []))

  #   return examples

  def _get_useracts(
      self, last_frame_json: Optional[Json], frame_json: Json
  ) -> tuple[list[Action], dict[Action, str]]:
    """Gets all user actions for a given frame."""
    # Frames need to be from the same service
    if last_frame_json:
      assert last_frame_json['service'] == frame_json['service']
    service = frame_json['service']
    ret = []
    useract_descs = {}
    other_acts = ordered_set.OrderedSet()

    for action_json in frame_json['actions']:
      act = action_json['act']
      other_acts.add(act)

    # OFFER_INTENT from previous system turn
    last_offer_intent_act_json = None
    if last_frame_json:
      for action_json in last_frame_json['actions']:
        if action_json['act'] == 'OFFER_INTENT':
          assert last_offer_intent_act_json is None
          last_offer_intent_act_json = action_json

    for action_json in frame_json['actions']:
      act = action_json['act']
      if act == 'INFORM_INTENT':
        intent = action_json['values']
        assert len(intent) == 1
        intent = intent[0]
        intent_desc = self._intent_descs[service][intent]
        a = (act, intent)
        ret.append(a)
        useract_descs[a] = f'user wants to {intent_desc}'
      elif act == 'NEGATE_INTENT':
        assert last_offer_intent_act_json is not None
        offer_intent = last_offer_intent_act_json['values']
        assert len(offer_intent) == 1
        offer_intent = offer_intent[0]
        offer_intent_desc = self._intent_descs[service][offer_intent]
        a = (act, offer_intent)
        ret.append(a)
        useract_descs[a] = f'user no longer wants to {offer_intent_desc}'
      elif act == 'AFFIRM_INTENT':
        assert last_offer_intent_act_json is not None
        offer_intent = last_offer_intent_act_json['values']
        assert len(offer_intent) == 1
        offer_intent = offer_intent[0]
        offer_intent_desc = self._intent_descs[service][offer_intent]
        # AFFIRM_INTENT -> INFORM_INTENT
        a = (act, offer_intent)
        ret.append(a)
        useract_descs[a] = f'user wants to {offer_intent_desc}'
      elif act == 'INFORM':
        slot = action_json['slot']
        a = (act, slot)
        ret.append(a)
        # not formatted! to be formatted when slot_ord is known
        useract_descs[a] = 'user is informing {slot_ind}'
      elif act == 'REQUEST':
        slot = action_json['slot']
        a = (act, slot)
        ret.append(a)
        # not formatted! to be formatted when slot_ord is known
        useract_descs[a] = 'user is requesting {slot_ind}'
      elif act == 'AFFIRM':
        a = ('AFFIRM', None)
        ret.append(a)
        useract_descs[a] = 'user is confirming slot values'
      elif act == 'NEGATE':
        if 'THANK_YOU' in other_acts or 'GOODBYE' in other_acts:
          continue
        a = (act, None)
        ret.append(a)
        useract_descs[a] = "user doesn't want to confirm what's been suggested"
      elif act == 'SELECT':
        a = (act, None)  # common to all services
        ret.append(a)
        useract_descs[a] = 'user is selecting something you offered'
      elif act == 'REQUEST_ALTS':
        a = (act, None)  # common to all services
        ret.append(a)
        useract_descs[a] = 'user would like some alternative options'
      elif act == 'THANK_YOU':
        a = (act, None)  # common to all services
        ret.append(a)
        useract_descs[a] = 'user is saying thanks'
      elif act == 'GOODBYE':
        a = (act, None)  # common to all services
        ret.append(a)
        useract_descs[a] = 'user is saying goodbye'
      else:
        act = action_json['act']
        raise ValueError(f'Unknown action {act}')
    return ret, useract_descs

  def _get_sysacts(
      self, frame_json: Json
  ) -> tuple[list[Action], dict[Action, str]]:
    """Gets all system acts. Currently unused."""
    service = frame_json['service']
    ret = []
    sysact_descs = {}
    other_acts = ordered_set.OrderedSet()

    for action_json in frame_json['actions']:
      act = action_json['act']
      other_acts.add(act)

    for action_json in frame_json['actions']:
      act = action_json['act']
      if act == 'INFORM':
        slot = action_json['slot']
        a = (act, slot)
        ret.append(a)
        # not formatted! to be formatted when slot_ord is known
        sysact_descs[a] = 'inform the user of {slot_ind}'
      elif act == 'REQUEST':
        slot = action_json['slot']
        a = (act, slot)
        ret.append(a)
        # not formatted! to be formatted when slot_ord is known
        sysact_descs[a] = 'request {slot_ind} from the user'
      elif act == 'CONFIRM':
        slot = action_json['slot']
        a = (act, slot)
        ret.append(a)
        sysact_descs[a] = 'confirm value of {slot_ind}'
      elif act == 'OFFER':
        slot = action_json['slot']
        a = (act, slot)
        ret.append(a)
        sysact_descs[a] = 'offer a value for {slot_ind}'
      elif act == 'NOTIFY_SUCCESS':
        a = (act, None)  # common to all services
        ret.append(a)
        sysact_descs[a] = 'inform the user their request succeeded'
      elif act == 'NOTIFY_FAILURE':
        a = (act, None)  # common to all services
        ret.append(a)
        sysact_descs[a] = 'inform the user their request failed'
      elif act == 'INFORM_COUNT':
        a = (act, None)
        ret.append(a)
        sysact_descs[a] = (
            'inform user the number of things that match their params'
        )
      elif act == 'OFFER_INTENT':
        intent = action_json['values']
        assert len(intent) == 1
        intent = intent[0]
        intent_desc = self._intent_descs[service][intent]
        a = (act, intent)
        ret.append(a)
        sysact_descs[a] = f"ask the user if they'd like to {intent_desc}"
      elif act == 'REQ_MORE':
        a = (act, None)  # common to all services
        ret.append(a)
        sysact_descs[a] = "ask the user if there's anything else they need"
      elif act == 'GOODBYE':
        a = (act, None)  # common to all services
        ret.append(a)
        sysact_descs[a] = 'say goodbye to the user'
      else:
        act = action_json['act']
        raise ValueError(f'Unknown action {act}')
    return ret, sysact_descs

  def _mode_is_zombie(self) -> bool:
    return 'zombie' in self._mode.value

  def _mode_is_zombie_hist(self) -> bool:
    return 'zombie_history' in self._mode.value

  def _build_acts(self):
    """Gather all actions across all frames."""
    frame_useracts = collections.defaultdict(ordered_set.OrderedSet)
    frame_useract_descs = collections.defaultdict(dict)
    frame_sysacts = collections.defaultdict(ordered_set.OrderedSet)
    frame_sysact_descs = collections.defaultdict(dict)
    for dialog_json in self._dialog_jsons:
      last_frame_jsons = {}
      for _, turn_json in enumerate(dialog_json['turns']):
        if turn_json['speaker'] == 'USER':
          for _, frame_json in enumerate(turn_json['frames']):
            service = frame_json['service']
            last_frame_json = last_frame_jsons.get(service, None)
            useracts, useract_descs = self._get_useracts(
                last_frame_json, frame_json
            )
            frame_useracts[service] |= ordered_set.OrderedSet(useracts)
            frame_useract_descs[service].update(useract_descs)
        elif turn_json['speaker'] == 'SYSTEM':
          for _, frame_json in enumerate(turn_json['frames']):
            service = frame_json['service']
            sysacts, sysact_descs = self._get_sysacts(frame_json)
            frame_sysacts[service] |= ordered_set.OrderedSet(sysacts)
            frame_sysact_descs[service].update(sysact_descs)
        for _, frame_json in enumerate(turn_json['frames']):
          service = frame_json['service']
          last_frame_jsons[service] = frame_json

    if self._mode_is_zombie():
      for service in frame_sysacts:
        frame_sysacts[service].add(('query', None))
        frame_sysact_descs[service][('query', None)] = '{QUERY}'
        frame_sysacts[service].add(('query_book', None))
        frame_sysact_descs[service][('query_book', None)] = '{QUERY_BOOK}'
        frame_sysacts[service].add(('query_check', None))
        frame_sysact_descs[service][('query_check', None)] = '{QUERY_CHECK}'

    self._frame_useracts = frame_useracts
    self._frame_useract_descs = frame_useract_descs
    self._frame_sysacts = frame_sysacts
    self._frame_sysact_descs = frame_sysact_descs

  # TODO(jeffreyzhao): Unused code for inferring policy graph.
  # def _build_graph(self):
  #   """Infer the policy graph from all convos in a service."""
  #   service_graph = collections.defaultdict(
  #       lambda: collections.defaultdict(list))
  #   for dialog_json in self._dialog_jsons:
  #     frame_json_hist = collections.defaultdict(dict)
  #     for turn, turn_json in enumerate(dialog_json['turns']):
  #       if turn_json['speaker'] == 'SYSTEM':
  #         for _, frame_json in enumerate(turn_json['frames']):
  #           service = frame_json['service']
  #           # All convos start with user turn
  #           last_user_frame_json = frame_json_hist[turn - 1].get(
  #               service, None)
  #           last_sys_frame_json = frame_json_hist.get(
  #               turn - 2, {}).get(service, None)
  #           useracts, _ = self._get_useracts(last_sys_frame_json,
  #                                            last_user_frame_json)
  #           sysacts, _ = self._get_sysacts(frame_json)

  #           big_useracts = ordered_set.OrderedSet(
  #               [useract[0] for useract in useracts])
  #           big_sysacts = ordered_set.OrderedSet(
  #               [sysact[0] for sysact in sysacts])
  #           # print(big_useracts, big_sysacts)
  #           # service_graph[service][] =

  #       for _, frame_json in enumerate(turn_json['frames']):
  #         service = frame_json['service']
  #         frame_json_hist[turn][service] = frame_json
  #   self._service_graph = service_graph

  def _convert_anytodog(self, dialog_json: Json) -> list[Example]:
    """Converts a single dialog JSON into per-turn per-frame examples."""
    examples = []
    convo_hist = []
    frame_act_hist = collections.defaultdict(list)
    # turn -> service -> frame_json
    frame_json_hist = collections.defaultdict(dict)

    def _is_categ_slot(service, slot):
      if not self._use_cat_slots:
        return False
      if service == 'Restaurants_1' and slot == 'cuisine':
        # Has a lot of values that aren't listed in schema.
        return False
      poss_vals = self._slot_cat_vals[service][slot]
      return poss_vals and not all([v.isdigit() for v in poss_vals])

    for turn, turn_json in enumerate(dialog_json['turns']):
      speaker, utt = turn_json['speaker'], turn_json['utterance']
      convo_str = ' '.join(  # pylint: disable=unused-variable
          f'[ {speaker} ] {utt}' for speaker, utt in convo_hist
      )

      if speaker == 'SYSTEM':
        for frame, frame_json in enumerate(turn_json['frames']):
          service = frame_json['service']
          # All convos start with user turn
          last_user_frame_json = frame_json_hist[turn - 1].get(service, None)
          last_sys_frame_json = frame_json_hist.get(turn - 2, {}).get(
              service, None
          )

          bs = {}
          for slot, val in last_user_frame_json['state']['slot_values'].items():
            bs[slot] = val[0]
          all_slots = list(self._slot_descs[service].keys())
          slot_ord = Ordering(all_slots, self._shuffle)
          cat_val_ord = {
              slot: Ordering([cv.lower() for cv in cat_vals], self._shuffle)
              for slot, cat_vals in self._slot_cat_vals[service].items()
          }
          bs_src = []
          bs_tgt = []
          for i, slot in slot_ord:
            slot_desc = self._slot_descs[service][slot]
            bs_src_piece = [f'p{i}={slot_desc}']

            if _is_categ_slot(service, slot):
              for j, cat_val in cat_val_ord[slot]:
                bs_src_piece.append(f'{j}) {cat_val}')
            bs_src.append(' '.join(bs_src_piece))

            val = bs.get(slot, None)
            if val:
              val = val.lower()
              if _is_categ_slot(service, slot):
                if val == 'dontcare':
                  # Allow dontcare as a value for categorical slots.
                  # TODO(jeffreyzhao): May need to reconsider this.
                  continue
                elif val.lower() in cat_val_ord[slot]:
                  val_ind = cat_val_ord[slot].get_idx(val)
                  val = string.ascii_letters[val_ind]
                else:
                  print(
                      f'SKIPPING CATEGORICAL SLOT {service} {slot} {val} '
                      f'{cat_val_ord[slot].tolist()}'
                  )
                  continue
              bs_tgt.append(f'p{i}={val}')
          bs_src = '; '.join(bs_src)
          bs_tgt = '; '.join(bs_tgt)

          all_useracts = list(self._frame_useracts[service])
          useract_ord = Ordering(all_useracts, self._shuffle)
          all_sysacts = list(self._frame_sysacts[service])
          sysact_ord = Ordering(all_sysacts, self._shuffle)

          useract_src = []
          for i, useract in useract_ord:
            useract_desc = self._frame_useract_descs[service][useract]
            if useract[0] in ('INFORM', 'REQUEST'):
              slot_ind = slot_ord.get_idx(useract[1])
              useract_desc = useract_desc.format(slot_ind=f'p{slot_ind}')
            useract_src.append(f'u{i}={useract_desc}')
          useract_src = '; '.join(useract_src)
          tgt_useracts, _ = self._get_useracts(
              last_sys_frame_json, last_user_frame_json
          )
          if self._mode_is_zombie_hist():
            frame_act_hist[service].append(tgt_useracts)
            act_hist_inds = []
            for i, acts in enumerate(frame_act_hist[service]):
              if i % 2 == 0:
                useract_inds = [useract_ord.get_idx(ua) for ua in acts]
                useract_inds.sort()
                act_hist_inds.append(
                    ' '.join(f'u{ind}' for ind in useract_inds)
                )
              else:
                sysact_inds = [sysact_ord.get_idx(ua) for ua in acts]
                sysact_inds.sort()
                act_hist_inds.append(' '.join(f's{ind}' for ind in sysact_inds))
            act_hist_tgt = '; '.join(act_hist_inds)
          else:
            useract_tgt_inds = [useract_ord.get_idx(ua) for ua in tgt_useracts]
            useract_tgt_inds.sort()
            useract_tgt = '; '.join(f'u{i}' for i in useract_tgt_inds)

          sysact_src = []
          for i, sysact in sysact_ord:
            sysact_desc = self._frame_sysact_descs[service][sysact]
            if sysact[0] in ('INFORM', 'REQUEST', 'CONFIRM', 'OFFER'):
              slot_ind = slot_ord.get_idx(sysact[1])
              sysact_desc = sysact_desc.format(slot_ind=f'p{slot_ind}')
            sysact_src.append(f's{i}={sysact_desc}')
          tgt_sysacts, _ = self._get_sysacts(frame_json)
          sysact_tgt_inds = [sysact_ord.get_idx(ua) for ua in tgt_sysacts]
          sysact_tgt_inds.sort()
          sysact_src = '; '.join(sysact_src)
          sysact_tgt = ' '.join(f's{i}' for i in sysact_tgt_inds)

          # Generate recommended system actions.
          self._total_turns += 1

          gt_intent = last_user_frame_json['state']['active_intent']
          cur_intent = 'NONE'
          for i in range(len(frame_act_hist[service]) - 1, -1, -1):
            turn_acts = frame_act_hist[service][i]
            done = False
            for act in turn_acts:
              if act[0] in ['INFORM_INTENT', 'AFFIRM_INTENT']:
                cur_intent = act[1]
                done = True
              elif act[0] == 'NEGATE_INTENT':
                cur_intent = 'NONE'
                done = True
              elif (
                  # This should be NEGATE, but we filter out this action
                  act[0] in ('THANK_YOU', 'GOODBYE')
                  and i > 0
                  and any(
                      a[0] == 'REQ_MORE' for a in frame_act_hist[service][i - 1]
                  )
              ):
                # If user is doing NEGATE and last system turn had REQ_MORE
                # i.e. if the system asked if the user need anything else, and
                # they said no
                cur_intent = 'NONE'
                done = True
              if done:
                break
            if done:
              break
          assert cur_intent == gt_intent

          sys_act_recs = anytod_policy_function(
              bs,
              cur_intent,
              frame_act_hist[service],
              self._services[service],
          )
          if set(sys_act_recs).issuperset(set(tgt_sysacts)):
            self._total_turns_correct += 1
          sysact_recs_inds = [
              sysact_ord.get_idx(ua)
              for ua in sys_act_recs
              if sysact_ord.__contains__(ua)
          ]
          sysact_recs_inds.sort()
          sysact_recs_src = ' '.join(f's{i}' for i in sysact_recs_inds)

          if self._mode_is_zombie_hist():
            frame_act_hist[service].append(tgt_sysacts)

          act_hist_tag = '[ history ]' if self._fix_tags else '[ user actions ]'
          select_tag = '[ select ]' if self._fix_tags else '[ system actions ]'

          if _MODE.value == Mode.ZOMBIE_HISTORY_2PASS:
            inp = (
                f'[ params ] {bs_src} [ user actions ] {useract_src} '
                f'[ system actions ] {sysact_src} [ conversation ] {convo_str}'
            )
            val1 = f'[ belief state ] {bs_tgt} {act_hist_tag} {act_hist_tgt}'
            val2 = f'[ recommended actions ] {sysact_recs_src}'
            val3 = (
                f'{select_tag} {sysact_tgt} [ response ] ['
                f' {speaker.lower()} ] {utt}'
            )

            metadata = Json({
                'slot_ord': slot_ord.tolist(),
                'useract_ord': useract_ord.tolist(),
                'sysact_ord': sysact_ord.tolist(),
                'service': service,
                'turn': turn,
                'frame': frame,
            })
            examples.append(
                Example(
                    inp,
                    val1,
                    dialog_json['dialogue_id'],
                    turn,
                    frame,
                    service,
                    metadata,
                    [],
                )
            )
            examples.append(
                Example(
                    ' '.join([inp, val1, val2]),
                    val3,
                    dialog_json['dialogue_id'],
                    turn,
                    frame,
                    service,
                    metadata,
                    [],
                )
            )
          else:
            if _MODE.value == Mode.ZOMBIE:
              inp = (
                  f'[ params ] {bs_src} [ user actions ]  {useract_src} ['
                  f' system actions ] {sysact_src} [ conversation ] {convo_str}'
              )
              val = f'[ belief state ] {bs_tgt} {act_hist_tag} {useract_tgt}'
            elif _MODE.value == Mode.ZOMBIE_HISTORY:
              inp = (
                  f'[ params ] {bs_src} [ user actions ] {useract_src} '
                  f'[ system actions ] {sysact_src} [ conversation ] '
                  f'{convo_str}'
              )
              val = f'[ belief state ] {bs_tgt} {act_hist_tag} {act_hist_tgt}'
            elif _MODE.value == Mode.ZOMBIE_HISTORY_2PASS_2NDONLY:
              inp = (
                  f'[ params ] {bs_src} [ user actions ] {useract_src} '
                  f'[ system actions ] {sysact_src} [ conversation ] '
                  f'{convo_str} [ belief state ] {bs_tgt} {act_hist_tag} '
                  f'{act_hist_tgt} [ recommended actions ] {sysact_recs_src}'
              )
              val = (
                  f'{select_tag} {sysact_tgt} [ response ] '
                  f'[ {speaker.lower()} ] {utt}'
              )
            elif _MODE.value == Mode.ZOMBIE_HISTORY_2PASS_EVAL:
              inp = (
                  f'[ params ] {bs_src} [ user actions ] {useract_src} '
                  f'[ system actions ] {sysact_src} [ conversation ] '
                  f'{convo_str} '
              )
              val = (
                  f'[ belief state ] {bs_tgt} {act_hist_tag} '
                  f'{act_hist_tgt} [ recommended actions ] {sysact_recs_src} '
                  f'{select_tag} {sysact_tgt} [ response ] '
                  f'[ {speaker.lower()} ] {utt}'
              )
            metadata = Json({
                'slot_ord': slot_ord.tolist(),
                'useract_ord': useract_ord.tolist(),
                'sysact_ord': sysact_ord.tolist(),
                'service': service,
                'turn': turn,
                'frame': frame,
                'dialog_id': dialog_json['dialogue_id'],
                'policy': 'sgd',
            })
            examples.append(
                Example(
                    inp,
                    val,
                    dialog_json['dialogue_id'],
                    turn,
                    frame,
                    service,
                    metadata,
                    [],
                )
            )

      convo_hist.append((speaker, utt))
      for frame, frame_json in enumerate(turn_json['frames']):
        service = frame_json['service']
        frame_json_hist[turn][service] = frame_json

    return examples

  def convert_to_anytodog_format(self) -> list[Example]:
    """Top-level function to convert data to AnyTOD format."""
    ret = []
    for d in self._dialog_jsons:
      ret.extend(self._convert_anytodog(d))

    print(
        '%d out of %d turns had the recommended system actions be a'
        ' superset of the actual system actions.'
        % (self._total_turns_correct, self._total_turns)
    )

    return ret


def main(_):
  split_to_exs = {}
  for split in ['train', 'dev', 'test']:
    dialogs, schema = load_sgd_data(os.path.join(_INPUT_DIR.value, split))
    converter = Converter(
        dialogs,
        schema,
        _MODE.value,
        _SHUFFLE.value,
        _CAT_SLOTS.value,
        _FIX_TAGS.value,
    )
    examples = converter.convert_to_anytodog_format()
    if _SHUFFLE.value and split == 'train':
      random.shuffle(examples)
    split_to_exs[split] = examples

  split_to_exs['devtest'] = split_to_exs['dev'] + split_to_exs['test']
  split_to_exs['devsample'] = split_to_exs['dev'][:2048]

  tf.io.gfile.makedirs(_OUTPUT_DIR.value)
  for split, exs in split_to_exs.items():
    with tf.io.TFRecordWriter(
        os.path.join(_OUTPUT_DIR.value, f'{split}.tfrecord')
    ) as rw:
      for ex in exs:
        rw.write(ex.build_tf_example().SerializeToString())
      print(f'Write {len(exs)} examples to {split}')


if __name__ == '__main__':
  app.run(main)
