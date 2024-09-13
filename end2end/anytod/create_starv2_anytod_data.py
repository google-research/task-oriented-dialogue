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

"""Create STARv2 AnyTOD data."""
import collections
import copy
import dataclasses
import json
import os
import random
import string
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from task_oriented_dialogue.end2end.anytod import starv2_lib
import tensorflow as tf

_DATADIR = flags.DEFINE_string(
    'data_dir',
    '',
    'Input STAR datadir.',
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir', '', 'Output AnyTOD directory to write records.'
)
_USE_CATEGORICAL = flags.DEFINE_boolean(
    'use_cat_slots', True, 'Whether to use categorical slots,'
)
_USE_REPLIES = flags.DEFINE_boolean(
    'use_replies_as_desc',
    False,
    'Whether to use template utterances as descriptions.',
)
_USE_GUIDANCE = flags.DEFINE_boolean(
    'use_guidance', True, 'Whether to have the recommended action guidance.'
)
_FULLSHOT_PERCENT = flags.DEFINE_float(
    'fullshot_percent', 0.8, 'Percentage of data for fullshot exps.'
)
_FIX_TAGS = flags.DEFINE_bool('fix_tags', False, 'If true, use fixed tags.')

TASKS = [
    'apartment_schedule',
    'apartment_search',
    'bank_balance',
    'bank_fraud_report',
    'doctor_followup',
    'doctor_schedule',
    'hotel_book',
    'hotel_search',
    'hotel_service_request',
    'meeting_schedule',
    'party_plan',
    'party_rsvp',
    'plane_book',
    'plane_search',
    'restaurant_book',
    'restaurant_search',
    'ride_book',
    'ride_change',
    'ride_status',
    'spaceship_access_codes',
    'spaceship_life_support',
    'trip_directions',
    'trivia',
    'weather',
]


@dataclasses.dataclass
class Example:
  """Example dataclass."""

  dialog_id: starv2_lib.DialogId
  turn: int
  inp: str
  val: str
  metadata: starv2_lib.Json

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
                'value': _bytes_feature(self.val.encode('utf-8')),
                'dialog_id': _bytes_feature(
                    str(self.dialog_id).encode('utf-8')
                ),
                'turn': _int64_feature(self.turn),
                'metadata': _bytes_feature(
                    json.dumps(self.metadata).encode('utf-8')
                ),
                # Legacy field, keep this in
                'policy_table': _bytes_feature('[]'.encode('utf-8')),
            }
        )
    )


def _is_categorical_slot(param):
  if not _USE_CATEGORICAL.value:
    return False
  return param.type == 'Categorical' and len(param.categories) < 10


def _clean_bs(bs):
  new_bs = {}
  for slot, val in bs.items():
    # TODO(jeffreyzhao): Skip on dontcare!!! should be in annotated
    if val == 'dontcare':
      logging.warning('%s is dontcare!', slot)
      continue
    elif val == 'forgot':
      continue
    new_bs[slot] = val
  return new_bs


def _build_bs_tgt(
    bs: dict[str, str],
    api: starv2_lib.StarApi,
    slot_ord: starv2_lib.Ordering[str],
    cat_value_ord: dict[str, starv2_lib.Ordering[str]],
):
  """Build the belief state target string."""
  bs_tgt = []
  for slot_ind, slot in slot_ord:
    val = bs.get(slot, None)
    assert val != 'dontcare'
    param = api.params_by_name[slot]
    if val:
      if _is_categorical_slot(param):
        # assert val in cat_value_ord[slot], val
        if val not in cat_value_ord[slot]:
          logging.warning('not in cat val: %s %s', slot, val)
          continue
        val_ind = cat_value_ord[slot].get_idx(val)
        val = string.ascii_letters[val_ind]
      bs_tgt.append(f'p{slot_ind}={val}')
  bs_tgt = '; '.join(bs_tgt)
  return bs_tgt


def generate_examples(
    data: starv2_lib.StarData,
) -> dict[tuple[starv2_lib.DialogId, str], list[Example]]:
  """Main function for generating STARv2 AnyTOD examples."""

  # TODO(jeffreyzhao): Refactor starv2_lib to avoid protected access.
  def _extract_wizard_task_info(dialog):
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

  def _get_sysact(event):
    if starv2_lib.is_wizard_event(event):
      if 'ActionLabel' in event:
        sysact_tgt = event['ActionLabel']
      else:
        sysact_tgt = starv2_lib.WIZARD_CUSTOM_LABEL
    else:
      sysact_tgt = starv2_lib.get_query_event_act(event)
    return sysact_tgt

  num_sysacts_not_rec = collections.defaultdict(lambda: [0, 0])
  sysacts_not_rec = collections.defaultdict(collections.Counter)
  num_train_exs_with_sysact = collections.defaultdict(
      lambda: collections.defaultdict(int)
  )
  # (task, split) -> list[examples]
  # split_exs = collections.defaultdict(list)
  all_exs = collections.defaultdict(list)
  for task in TASKS:
    api = data.task_to_api[task]
    graph = data.task_to_graph[task]
    slots = [p.name for p in api.params]
    useract_descs = graph.json['useract_descs']
    sysact_descs = graph.json['sysact_descs']
    # TODO(jeffreyzhao): user custom act ok?
    useracts = list(useract_descs.keys()) + [starv2_lib.USER_CUSTOM_LABEL]
    sysacts = list(sysact_descs.keys()) + [starv2_lib.OUT_OF_SCOPE_LABEL]

    for dialog_id in data.task_to_ids[task]:
      dialog = data.dialogs[dialog_id]
      dialog_json = dialog.json
      for turn, event in enumerate(dialog_json['Events']):
        if starv2_lib.is_wizard_or_query_event(event):
          is_bad_user_label = False
          is_wiz_custom_act = False
          hist_has_wiz_custom_act = False

          # Construct orderings
          slot_ord = starv2_lib.Ordering(slots, True)
          useract_ord = starv2_lib.Ordering(useracts, True)
          sysact_ord = starv2_lib.Ordering(sysacts, True)
          cat_value_ord = {}
          for p in api.params:
            if _is_categorical_slot(p):
              cat_value_ord[p.name] = starv2_lib.Ordering(
                  [v.lower() for v in p.categories], True
              )

          # Construct action descriptions
          if _USE_REPLIES.value:
            for useract in useracts:
              if useract == starv2_lib.USER_CUSTOM_LABEL:
                useract_descs[useract] = 'user custom action'
              else:
                useract_descs[useract] = graph.replies[useract]
            for sysact in sysacts:
              if sysact == starv2_lib.OUT_OF_SCOPE_LABEL:
                sysact_descs[sysact] = (
                    "tell user you don't understand what they want"
                )
              else:
                sysact_descs[sysact] = graph.replies[sysact]
          else:
            for useract in useracts:
              # If this is a user informing act, replace with slot ind
              slot = graph.user_inform_act_to_slot.get(useract, None)
              if slot is None:
                continue
              slot_ind = slot_ord.get_idx(slot)
              useract_descs[useract] = f'user is informing p{slot_ind}'
            useract_descs[starv2_lib.USER_CUSTOM_LABEL] = (
                'user is doing something out of scope'
            )
            for sysact in sysacts:
              # If this is a system requesting act, replace with slot ind
              slot = graph.ask_act_to_slot_name.get(sysact, None)
              if slot is None:
                continue
              slot_ind = slot_ord.get_idx(slot)
              sysact_descs[sysact] = f'request p{slot_ind} from the user'
            sysact_descs[starv2_lib.OUT_OF_SCOPE_LABEL] = (
                "tell user you don't understand what they want"
            )

          # Build convo str, include up to last user turn
          convo_hist = []
          convo_hist_bs = {}
          for hist_event in dialog_json['Events'][: turn + 1]:
            if starv2_lib.is_relevant_event(hist_event):
              # Prepend wiz info to first utt
              prepend = ''
              if not convo_hist:
                wiz_info = _extract_wizard_task_info(dialog)
                # TODO(jeffreyzhao): hack!!!
                prepend = '. '.join(wiz_info) + '. '
              if starv2_lib.is_user_event(hist_event):
                convo_hist_bs = _clean_bs(hist_event['PredictedBeliefState'])
              if starv2_lib.is_query_event(hist_event):
                text = '[ query ] ' + _build_bs_tgt(
                    convo_hist_bs, api, slot_ord, cat_value_ord
                )
              else:
                text = dialog.build_event_str(hist_event, slot_ord, prepend)
              convo_hist.append(text)
          response = convo_hist[-1]
          convo_hist_list = copy.deepcopy(convo_hist)
          convo_hist = ' '.join(convo_hist[:-1])

          # Build params ctx str
          name_to_param = {p.name: p for p in api.params}
          slot_ctx = []
          for slot_ind, slot in slot_ord:
            param = name_to_param[slot]
            slot_desc = param.readable_name
            pieces = [f'p{slot_ind}={slot_desc}']
            if _is_categorical_slot(param):
              for val_ind, val in cat_value_ord[slot]:
                val_let = string.ascii_letters[val_ind]
                pieces.append(f'{val_let}) {val}')
            slot_ctx.append(' '.join(pieces))
          slot_ctx = '; '.join(slot_ctx)

          assert all(ua_desc for ua_desc in useract_descs.values())
          assert all(sa_desc for sa_desc in sysact_descs.values())

          # Build useract, sysact ctx str
          useract_ctx = '; '.join(
              f'u{i}={useract_descs[ua]}' for i, ua in useract_ord
          )
          sysact_ctx = '; '.join(
              f's{i}={sysact_descs[sa]}' for i, sa in sysact_ord
          )

          user_turn, user_event = dialog.get_closest_event(
              turn, starv2_lib.is_user_or_result_event, reverse=True
          )
          assert user_event

          bs = _clean_bs(user_event['PredictedBeliefState'])
          bs_tgt = _build_bs_tgt(bs, api, slot_ord, cat_value_ord)

          # Construct action history target
          # each turn has multiple user acts
          # turns are separated by ;
          # e.g. u1 u2 u3 ; s3 s5 ; u3 u5 ; s2 ; ...
          act_hist = []
          useract_tgt = []
          for hist_event in dialog_json['Events'][:turn]:
            if starv2_lib.is_user_or_result_event(hist_event):
              turn_useract_tgt = []
              act_hist.append(hist_event['ActionLabel'])
              for useract in hist_event['ActionLabel']:
                # If any action in the history has bad label
                if useract in (
                    starv2_lib.USER_BAD_LABEL,
                    starv2_lib.RESULT_BAD_LABEL,
                ):
                  is_bad_user_label = True
                  turn_useract_tgt.append(999)
                else:
                  turn_useract_tgt.append(useract_ord.get_idx(useract))
              useract_tgt.append(
                  ' '.join([f'u{i}' for i in sorted(turn_useract_tgt)])
              )

            elif starv2_lib.is_wizard_or_query_event(hist_event):
              sysact = _get_sysact(hist_event)
              act_hist.append(sysact)
              if sysact == starv2_lib.WIZARD_CUSTOM_LABEL:
                hist_has_wiz_custom_act = True
                sysact_ind = 999
              else:
                sysact_ind = sysact_ord.get_idx(sysact)
              useract_tgt.append(f's{sysact_ind}')
          useract_tgt = '; '.join(useract_tgt)

          sysact_tgt = _get_sysact(event)
          if sysact_tgt == starv2_lib.WIZARD_CUSTOM_LABEL:
            is_wiz_custom_act = True

          if is_wiz_custom_act:
            continue

          # Get recommended system acts according to policy
          primary_item = event.get('PrimaryItem', {})
          rec_sysacts = starv2_lib.policy_with_hist(
              bs,
              act_hist,
              task,
              data.task_to_api[task],
              data.task_to_graph[task],
              convo_hist_list,
              primary_item,
          )
          not_rec = False
          if not is_bad_user_label:
            # Was the ground truth system action not in recommendations?
            if sysact_tgt not in rec_sysacts:
              not_rec = True
              sysacts_not_rec[task][sysact_tgt] += 1
            num_sysacts_not_rec[task][1] += 1

          rec_sysacts = [f's{sysact_ord.get_idx(sa)}' for sa in rec_sysacts]
          rec_sysacts = ' '.join(rec_sysacts)

          act_hist_tag = (
              '[ history ]' if _FIX_TAGS.value else '[ user actions ]'
          )
          select_tag = '[ select ]' if _FIX_TAGS.value else '[ system actions ]'

          # Construct strings
          sysact_tgt_name = sysact_tgt
          sysact_tgt = f's{sysact_ord.get_idx(sysact_tgt)}'
          inp = (
              f'[ params ] {slot_ctx} [ user actions ] {useract_ctx} '
              f'[ system actions ] {sysact_ctx} [ conversation ] {convo_hist}'
          )
          tgt1 = f'[ belief state ] {bs_tgt} {act_hist_tag} {useract_tgt}'
          if _USE_GUIDANCE.value:
            tgt2 = f'[ recommended actions ] {rec_sysacts}'
          else:
            tgt2 = ''
          tgt3 = f'{select_tag} {sysact_tgt} [ response ] {response}'

          metadata = {
              'slot_ord': slot_ord.idx_to_entity,
              'useract_ord': useract_ord.idx_to_entity,
              'sysact_ord': sysact_ord.idx_to_entity,
              'task': task,
              'dialog_id': str(dialog.dialog_id),
              'turn': turn,
              'primary_item': primary_item,
              'convo_hist': convo_hist_list,
              'policy': 'starv2',
          }

          if not_rec:
            num_sysacts_not_rec[task][0] += 1

          # Skip bad labels in action history for training data
          if not (
              is_bad_user_label or hist_has_wiz_custom_act or is_wiz_custom_act
          ):
            if not _USE_GUIDANCE.value:
              ex = Example(
                  dialog_id, turn, inp, ' '.join([tgt1, tgt2, tgt3]), metadata
              )
              all_exs[(dialog_id, 'train')].append(ex)
            else:
              ex1 = Example(dialog_id, user_turn, inp, tgt1, metadata)
              ex2 = Example(
                  dialog_id, turn, ' '.join([inp, tgt1, tgt2]), tgt3, metadata
              )
              all_exs[(dialog_id, 'train')].append(ex1)
              all_exs[(dialog_id, 'train')].append(ex2)
            num_train_exs_with_sysact[task][sysact_tgt_name] += 1

          # Skip custom next action prediction for evaluation
          # This is done in the STAR baselines
          if not is_wiz_custom_act:
            ex = Example(
                dialog_id, turn, inp, ' '.join([tgt1, tgt2, tgt3]), metadata
            )
            all_exs[(dialog_id, 'test')].append(ex)

  num_not_rec_all = 0
  num_tot_all = 0
  print('NEXT ACTION PREDICTION RECOMMENDATION VIOATIONS:')
  for task, (num_not_rec, num_tot) in num_sysacts_not_rec.items():
    print(
        f'\t {task}: {num_not_rec} not rec out of {num_tot}'
        f' ({num_not_rec/num_tot:.04f})'
    )
    print('\t -', sysacts_not_rec[task].most_common(3))
    num_not_rec_all += num_not_rec
    num_tot_all += num_tot
  print(f'\tTotal across all tasks: {num_not_rec_all / num_tot_all:.04f}')
  print()

  return all_exs


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  random.seed(123)
  options = starv2_lib.Options(
      starv2_lib.ExampleFormat.TRANSITIONS_ANYTOD, False, False
  )
  starv2_lib.set_star_version(starv2_lib.StarVersion.V1)
  data = starv2_lib.load_star_jsons(_DATADIR.value, options)
  all_exs = generate_examples(data)

  def _get_domain(task):
    return task.split('_')[0]

  # Generate zero-shot domain data
  print('ZERO-SHOT DOMAIN')
  domains = set(_get_domain(task) for task in TASKS)
  for eval_domain in domains:
    train_exs = []
    test_exs = []
    for dialog_id, dialog in data.dialogs.items():
      task = dialog.task
      if _get_domain(task) == eval_domain:
        test_exs.extend(all_exs[(dialog_id, 'test')])
      else:
        train_exs.extend(all_exs[(dialog_id, 'train')])
    # Print out task, split, example count
    print(eval_domain, 'train', len(train_exs))
    print(eval_domain, 'test', len(test_exs))
    random.shuffle(train_exs)
    domain_dir = os.path.join(_OUTPUT_DIR.value, eval_domain)
    tf.io.gfile.makedirs(domain_dir)
    with tf.io.TFRecordWriter(os.path.join(domain_dir, 'train.tfrecord')) as rw:
      for ex in train_exs:
        rw.write(ex.build_tf_example().SerializeToString())
    with tf.io.TFRecordWriter(os.path.join(domain_dir, 'test.tfrecord')) as rw:
      for ex in test_exs:
        rw.write(ex.build_tf_example().SerializeToString())
  print()

  # Generate fullshot data
  print('FULLSHOT')
  all_dialog_ids = sorted(list(set(data.dialogs.keys())))
  fs_train_dialog_ids = all_dialog_ids[
      : int(len(all_dialog_ids) * _FULLSHOT_PERCENT.value)
  ]
  fs_test_dialog_ids = all_dialog_ids[
      int(len(all_dialog_ids) * _FULLSHOT_PERCENT.value) :
  ]
  for split, split_dialog_ids in [
      ('train', fs_train_dialog_ids),
      ('test', fs_test_dialog_ids),
  ]:
    exs = []
    for dialog_id in split_dialog_ids:
      dialog = data.dialogs[dialog_id]
      exs.extend(all_exs[(dialog_id, split)])
    if split == 'train':
      random.shuffle(exs)
    fullshot_dir = os.path.join(_OUTPUT_DIR.value, 'fullshot')
    tf.io.gfile.makedirs(fullshot_dir)
    with tf.io.TFRecordWriter(
        os.path.join(fullshot_dir, f'{split}.tfrecord')
    ) as rw:
      for ex in exs:
        rw.write(ex.build_tf_example().SerializeToString())
    print(f'fullshot {split} {len(exs)}')


if __name__ == '__main__':
  app.run(main)
