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

"""Common lib for STARv2 data manipulation."""
# TODO(jeffreyzhao): This is research code and still being experimented with!
# Need to clean up once in more stable state.
import collections
import copy
import dataclasses
import enum
import functools
import json
import os
import random
import time
from typing import Callable, Generic, Iterator, Optional, Sequence, TypeVar

import tensorflow as tf

DialogId = int
Json = dict
Event = Json
BeliefState = dict[str, str]
ActionHistory = list[list[str]]
UserLabels = dict[int, str]
# TODO(jeffreyzhao): Should use these type annotations
Action = str
Trans = tuple[str, str]

# Globals for keeping track of dataset wide info.
# TODO(jeffreyzhao): This can be cleaned up.
ALL_ASK_ACTS = set()
# Number of target wizard actions that are not in the recommended policy actions
NUM_NOT_IN_GRAPH_TRAIN = 0
NUM_NOT_IN_GRAPH_TEST = 0


# example format
# actions
# [params] ... [actions] s1, s2, ... u1, u2, ..., q1, q2 [convo]
# [history] s1 u2 s2 u1
#
# transitions
# [params] ... [actions] t1 t2 [convo]
# [history] t2 t1
class ExampleFormat(enum.Enum):
  ACTIONS = 'actions'
  TRANSITIONS = 'transitions'
  TRANSITIONS_WIZ2WIZ = 'transitions_wiz2wiz'
  TRANSITIONS_GRAPH_WIZ2WIZ = 'transitions_graph_wiz2wiz'
  ACTIONS_ANYTOD = 'actions_anytod'
  TRANSITIONS_ANYTOD = 'transitions_anytod'


QUERY_LABEL = 'query'
RESULT_LABEL = 'result'
WIZARD_CUSTOM_LABEL = 'custom'
USER_CUSTOM_LABEL = 'user_custom'
USER_BAD_LABEL = 'user_bad_label'
RESULT_BAD_LABEL = 'result_bad_label'
RESULT_CUSTOM_LABEL = 'result_custom'
ENDING_ACTIONS = ('anything_else', 'goodbye_1', 'goodbye_2')
OUT_OF_SCOPE_LABEL = 'out_of_scope'

API_CALL_MARKER = '[ query ]'
API_RETURN_MARKER = '[ query result ]'
USER_MARKER = '[ user ]'
SYSTEM_MARKER = '[ system ]'
CONVERSATION_MARKER = '[ conversation ]'
BELIEF_STATE_MARKER = '[ belief state ]'
ACTION_HISTORY_MARKER = '[ action history ]'
RECOMMENDED_ACTIONS_MARKER = '[ recommended actions ]'
SELECTED_ACTION_MARKER = '[ selected action ]'

DST_PREDICT_EXAMPLE_TGT = 'PREDICT EXAMPLE'


class StarVersion(enum.Enum):
  V1 = 1
  V2 = 2


STAR_VERSION = StarVersion.V1


def set_star_version(v: StarVersion):
  global STAR_VERSION
  STAR_VERSION = v


def is_user_event(event):
  return event['Agent'] == 'User' and event['Action'] == 'utter'


def is_custom_wizard_event(event):
  return event['Action'] == 'utter' and event['Agent'] == 'Wizard'


def is_picked_wizard_event(event):
  return event['Action'] == 'pick_suggestion'


def is_wizard_event(event):
  return is_custom_wizard_event(event) or is_picked_wizard_event(event)


def is_query_event(event):
  return event['Action'] == 'query'


def is_wizard_or_query_event(event):
  return is_wizard_event(event) or is_query_event(event)


def is_result_event(event):
  return event['Action'] == 'return_item'


def is_user_or_result_event(event):
  return is_user_event(event) or is_result_event(event)


def is_relevant_event(event):
  return is_user_or_result_event(event) or is_wizard_or_query_event(event)


# pytype: disable=bad-return-type
def get_action_label(event) -> Action:
  """Infer action label without looking at JSON's 'ActionLabel' field."""
  assert not is_user_event(event)
  if is_query_event(event):
    # TODO(jeffreyzhao): Use _get_query_event_belief_state?
    return QUERY_LABEL
  elif is_result_event(event):
    return RESULT_LABEL
  elif is_picked_wizard_event(event):
    if STAR_VERSION == StarVersion.V1:
      return event['ActionLabel']
    elif STAR_VERSION == StarVersion.V2:
      return event['ActionLabel'][0]
  elif is_custom_wizard_event(event):
    return WIZARD_CUSTOM_LABEL
  else:
    raise ValueError(f'can\'t determine action label: {event}')


def get_action_label_from_json(event) -> Action:
  """Infer action label from JSON's 'ActionLabel' field."""
  if is_custom_wizard_event(event):
    return WIZARD_CUSTOM_LABEL
  elif isinstance(event['ActionLabel'], str):
    return event['ActionLabel']
  elif isinstance(event['ActionLabel'], list):
    if event['ActionLabel']:
      return event['ActionLabel'][0]
    else:
      if is_result_event(event):
        return RESULT_CUSTOM_LABEL
      elif is_wizard_event(event):
        # TODO(jeffreyzhao): should have result custom
        return USER_CUSTOM_LABEL
  else:
    raise ValueError(f'can\'t determine action label: {event}')


# pytype: enable=bad-return-type


def get_event_speaker(event):
  if is_wizard_event(event):
    return 'wizard'
  elif is_user_event(event):
    return 'user'
  else:
    raise ValueError(event)


def get_query_event_belief_state(event: Event) -> BeliefState:
  """Determines action label for query event."""
  assert is_query_event(event)
  belief_state = {}
  for c in event['Constraints']:
    belief_state.update(c)
  if 'RequestType' in belief_state:
    del belief_state['RequestType']
  return belief_state


def get_query_event_act(event: Event) -> str:
  """Determines action label for query event."""
  arg_to_value = {}
  for d in event['Constraints']:
    for k, v in d.items():
      arg_to_value[k] = v
  query_label = 'query'
  request_types = set([
      c['RequestType'].replace('"', '')
      for c in event['Constraints']
      if 'RequestType' in c
  ])
  assert len(request_types) <= 2
  if 'Check' in request_types:
    query_label = 'query_check'
  if 'Book' in request_types:
    query_label = 'query_book'
  return query_label


def user_inform_act(task):
  return f'user_{task}_inform'


def is_ask_action(act):
  return act in ALL_ASK_ACTS


# pylint: disable=missing-function-docstring

# STARv2 user annotation labeling functions
# TODO(jeffreyzhao): In hindsight a lot of these functions were likely not
# necessary.


def user_label_trip_directions(task, graph, prev_wiz_act, next_wiz_act):
  del task, graph
  if prev_wiz_act is None and next_wiz_act == 'hello':
    return 'user_hello'
  elif prev_wiz_act is None and is_ask_action(next_wiz_act):
    return 'user_trip_directions_inform'
  elif prev_wiz_act == 'hello' and is_ask_action(next_wiz_act):
    return 'user_trip_directions_inform'
  elif is_ask_action(prev_wiz_act):
    return 'user_trip_directions_inform'
  elif next_wiz_act in ('trip_instructions_done',
                        'trip_inform_last_step_and_done'):
    return 'user_trip_directions_continue'
  elif next_wiz_act == 'trip_inform_detailed_step':
    return 'user_trip_directions_no_continue'
  elif prev_wiz_act == 'trip_inform_detailed_step' and next_wiz_act == 'trip_inform_simple_step_ask_proceed':
    return 'user_trip_directions_continue'
  elif prev_wiz_act == 'trip_inform_simple_step_ask_proceed' and next_wiz_act == 'trip_inform_simple_step_ask_proceed':
    return 'user_trip_directions_continue'
  elif next_wiz_act == 'goodbye_1':
    return 'user_trip_directions_goodbye'
  elif next_wiz_act == 'anything_else':
    return 'user_trip_directions_thanks'
  return None


def user_label_apartment_schedule(task, graph, prev_wiz_act, next_wiz_act):
  del task, graph
  found_apartment_acts = ('apartment_inform_viewing_available',
                          'apartment_inform_viewing_unavailable')
  if prev_wiz_act is None and next_wiz_act == 'hello':
    return 'user_hello'
  elif prev_wiz_act is None and is_ask_action(next_wiz_act):
    return 'user_apartment_schedule_inform'
  elif prev_wiz_act == 'hello' and is_ask_action(next_wiz_act):
    return 'user_apartment_schedule_inform'
  elif is_ask_action(prev_wiz_act):
    return 'user_apartment_schedule_inform'
  elif next_wiz_act in found_apartment_acts:
    return 'user_apartment_schedule_inform'
  elif prev_wiz_act == 'apartment_inform_viewing_unavailable' and next_wiz_act in found_apartment_acts:
    return 'user_apartment_schedule_inform'
  elif next_wiz_act == 'apartment_inform_booking_successful':
    return 'user_apartment_confirm_booking'
  elif next_wiz_act == 'goodbye_2':
    return 'user_apartment_thanks'
  elif next_wiz_act == 'anything_else':
    return 'user_apartment_thanks'
  return None


def user_label_catchall(task, graph, prev_wiz_act, next_wiz_act):
  del prev_wiz_act
  prev_user_acts = graph.graph_reverse[next_wiz_act]
  prev_user_acts = [
      a
      for a in prev_user_acts
      if not a.startswith('db_') and a not in graph.all_user_inform_acts
  ]
  if is_ask_action(next_wiz_act) or next_wiz_act in (
      'weather_inform_forecast',
      'hotel_inform_service_request_successful',
      'hotel_inform_service_request_failed',
      'apartment_inform_nothing_found',
      'apartment_inform_search_result',
      'plane_inform_flight_details',
      'plane_inform_nothing_found',
      'ride_inform_changes_failed',
      'ride_inform_changes_successful',
      'hotel_inform_nothing_found',
  ):
    return user_inform_act(task)
  if next_wiz_act == WIZARD_CUSTOM_LABEL:
    return None
  if not prev_user_acts:
    print(
        '_user_label_catchall: no previous user acts for next_wiz_act %s',
        next_wiz_act,
    )
    return None
  if len(prev_user_acts) != 1:
    print(
        (
            '_user_label_catchall: multiple previous user acts for next+wiz_act'
            ' %s (prev_user_acts: %s)'
        ),
        next_wiz_act,
        prev_user_acts,
    )
    return None
  return prev_user_acts[0]


def user_label_bank_balance(task, graph, prev_wiz_act, next_wiz_act):
  del task, graph
  if prev_wiz_act is None and next_wiz_act == 'hello':
    return 'user_hello'
  elif prev_wiz_act is None and is_ask_action(next_wiz_act):
    return 'user_bank_balance_inform'
  elif prev_wiz_act == 'hello' and is_ask_action(next_wiz_act):
    return 'user_bank_balance_inform'
  elif is_ask_action(prev_wiz_act):
    return 'user_bank_balance_inform'
  # TODO(jeffreyzhao): Forgot acts if prev_wiz_act is None
  # (user provides all slot values at start and tells wizard they don't know
  # everyhting)
  elif next_wiz_act == 'bank_inform_balance':
    return 'user_bank_balance_inform'
  elif prev_wiz_act == 'bank_inform_balance' and next_wiz_act in ENDING_ACTIONS:
    return 'user_bank_thanks'
  elif prev_wiz_act == 'anything_else' and next_wiz_act == 'goodbye_1':
    return 'user_bank_nothing_else'
  elif next_wiz_act == 'goodbye_1':
    return 'user_bank_goodbye'
  return None


def user_label_bank_fraud_report(task, graph, prev_wiz_act, next_wiz_act):
  del task, graph
  if prev_wiz_act is None and next_wiz_act == 'hello':
    return 'user_hello'
  elif prev_wiz_act is None and is_ask_action(next_wiz_act):
    return 'user_bank_fraud_report_inform'
  elif prev_wiz_act == 'hello' and is_ask_action(next_wiz_act):
    return 'user_bank_fraud_report_inform'
  elif is_ask_action(prev_wiz_act):
    return 'user_bank_fraud_report_inform'
  # TODO(jeffreyzhao): Forgot acts if prev_wiz_act is None
  # (user provides all slot values at start and tells wizard they don't know
  # everyhting)
  elif next_wiz_act in ('bank_inform_fraud_report_submitted',
                        'bank_inform_cannot_authenticate'):
    return 'user_bank_fraud_report_inform'
  elif prev_wiz_act == 'bank_inform_cannot_authenticate' and next_wiz_act == 'anything_else':
    return 'user_bank_thanks'
  elif prev_wiz_act == 'bank_inform_fraud_report_submitted' and next_wiz_act == 'anything_else':
    return 'user_bank_thanks'
  elif prev_wiz_act == 'bank_inform_fraud_report_submitted' and next_wiz_act == 'goodbye_1':
    return 'user_bank_goodbye'
  elif prev_wiz_act == 'anything_else' and next_wiz_act == 'goodbye_1':
    return 'user_bank_nothing_else'
  elif next_wiz_act == 'goodbye_1':
    return 'user_bank_goodbye'
  return None


def user_label_doctor_followup(task, graph, prev_wiz_act, next_wiz_act):
  del task, graph
  if prev_wiz_act is None and next_wiz_act == 'hello':
    return 'user_hello'
  elif (prev_wiz_act == 'hello' or prev_wiz_act is None) and is_ask_action(
      next_wiz_act
  ):
    return 'user_doctor_followup_inform'
  elif is_ask_action(prev_wiz_act):
    return 'user_doctor_followup_inform'
  elif next_wiz_act == 'doctor_inform_doctors_instructions':
    return 'user_doctor_followup_inform'
  elif prev_wiz_act == 'anything_else' and next_wiz_act == 'goodbye_1':
    return 'user_doctor_nothing_else'
  elif next_wiz_act == 'goodbye_1':
    return 'user_doctor_followup_goodbye'
  elif prev_wiz_act == 'doctor_inform_doctors_instructions' and next_wiz_act in ENDING_ACTIONS:
    return 'user_doctor_thanks'
  return None


def user_label_doctor_schedule(task, graph, prev_wiz_act, next_wiz_act):
  del task, graph
  doctor_found_acts = ('doctor_inform_booking_available',
                       'doctor_inform_booking_unavailable')
  if prev_wiz_act is None and next_wiz_act == 'hello':
    return 'user_hello'
  elif prev_wiz_act is None and is_ask_action(next_wiz_act):
    return 'user_doctor_schedule_inform'
  elif prev_wiz_act == 'hello' and is_ask_action(next_wiz_act):
    return 'user_doctor_schedule_inform'
  elif is_ask_action(prev_wiz_act):
    return 'user_doctor_schedule_inform'
  elif prev_wiz_act is None and is_ask_action(next_wiz_act):
    return 'user_doctor_schedule_inform'
  elif next_wiz_act in doctor_found_acts:
    return 'user_doctor_schedule_inform_book'
  elif (prev_wiz_act == 'doctor_inform_booking_available' and
        next_wiz_act == 'doctor_inform_booking_successful'):
    return 'user_doctor_schedule_inform_book'
  elif (prev_wiz_act == 'doctor_inform_booking_successful' and
        next_wiz_act == 'anything_else'):
    return 'user_doctor_schedule_thanks'
  elif prev_wiz_act == 'anything_else' and next_wiz_act == 'goodbye_1':
    return 'user_doctor_nothing_else'
  elif next_wiz_act == 'goodbye_1':
    return 'user_doctor_schedule_goodbye'
  return None


def user_label_hotel_book(task, graph, prev_wiz_act, next_wiz_act):
  del task, graph
  reservation_actions = ('hotel_reservation_succeeded',
                         'hotel_reservation_failed', 'hotel_unavailable')
  if next_wiz_act == 'hello':
    return 'user_hello'
  elif prev_wiz_act == 'anything_else' and next_wiz_act == 'goodbye_1':
    return 'user_hotel_book_goodbye'
  elif prev_wiz_act in reservation_actions and next_wiz_act in ENDING_ACTIONS:
    return 'user_hotel_book_thank_you'
  elif prev_wiz_act == 'hotel_ask_confirm_booking' and next_wiz_act in reservation_actions:
    return 'user_hotel_book_inform_book'
  elif prev_wiz_act == 'hotel_ask_confirm_booking' and next_wiz_act == ENDING_ACTIONS:
    return 'user_hotel_book_inform_no_book'
  elif next_wiz_act in ('hotel_ask_confirm_booking', 'hotel_unavailable'):
    return 'user_hotel_book_inform'
  elif is_ask_action(prev_wiz_act):
    return 'user_hotel_book_inform'
  elif (prev_wiz_act == 'hello' or prev_wiz_act is None) and is_ask_action(
      next_wiz_act
  ):
    return 'user_hotel_book_inform'
  elif next_wiz_act == 'goodbye_1':
    return 'user_hotel_book_goodbye'
  return None


def user_label_meeting_schedule(task, graph, prev_wiz_act, next_wiz_act):
  del task, graph
  if prev_wiz_act == 'anything_else' and next_wiz_act == 'goodbye_1':
    return 'user_meeting_schedule_goodbye'
  elif (prev_wiz_act == 'meeting_inform_confirmed' and
        next_wiz_act in ENDING_ACTIONS):
    return 'user_meeting_schedule_thanks'
  elif next_wiz_act in ('meeting_inform_unavailable_ask_different_time',
                        'meeting_inform_confirmed'):
    return 'user_meeting_schedule_inform'
  elif is_ask_action(prev_wiz_act):
    return 'user_meeting_schedule_inform'
  elif prev_wiz_act is None and next_wiz_act == 'hello':
    return 'user_hello'
  elif (prev_wiz_act in ('hello', None)) and is_ask_action(next_wiz_act):
    return 'user_meeting_schedule_inform'
  return None


def user_label_party_plan(task, graph, prev_wiz_act, next_wiz_act):
  del task, graph
  if prev_wiz_act is None and next_wiz_act == 'hello':
    return 'user_hello'
  elif prev_wiz_act is None and is_ask_action(next_wiz_act):
    return 'user_party_plan_inform'
  elif prev_wiz_act == 'hello' and is_ask_action(next_wiz_act):
    return 'user_party_plan_inform'
  elif is_ask_action(prev_wiz_act):
    return 'user_party_plan_inform'
  elif next_wiz_act == 'party_ask_confirm_booking':
    return 'user_party_plan_inform'
  elif (prev_wiz_act == 'party_ask_confirm_booking' and
        next_wiz_act == 'party_booking_successful'):
    return 'user_party_plan_inform_book'
  elif (prev_wiz_act == 'party_booking_successful' and
        next_wiz_act == 'anything_else'):
    return 'user_party_plan_thanks'
  elif (prev_wiz_act == 'party_booking_successful' and
        next_wiz_act == 'goodbye_1'):
    return 'user_party_plan_thanks'
  elif next_wiz_act == 'party_ask_confirm_booking':
    return 'user_party_plan_inform'
  elif prev_wiz_act == 'anything_else' and next_wiz_act == 'goodbye_1':
    return 'user_party_plan_goodbye'
  elif prev_wiz_act == 'goodbye_1' and next_wiz_act == 'goodbye_1':
    return 'user_party_plan_goodbye'
  return None


def user_label_party_rsvp(task, graph, prev_wiz_act, next_wiz_act):
  del task, graph
  if prev_wiz_act is None and next_wiz_act == 'hello':
    return 'user_hello'
  elif prev_wiz_act is None and is_ask_action(next_wiz_act):
    return 'user_party_rsvp_inform'
  elif prev_wiz_act == 'hello' and is_ask_action(next_wiz_act):
    return 'user_party_rsvp_inform'
  elif is_ask_action(prev_wiz_act):
    return 'user_party_rsvp_inform'
  elif next_wiz_act == 'party_confirm_rsvp':
    return 'user_party_rsvp_inform'
  elif prev_wiz_act == 'party_confirm_rsvp' and next_wiz_act == 'party_confirm_rsvp':
    return 'user_party_rsvp_inform'
  elif prev_wiz_act == 'party_confirm_rsvp' and next_wiz_act == 'goodbye_1':
    return 'user_party_rsvp_goodbye'
  elif prev_wiz_act == 'party_confirm_rsvp' and next_wiz_act == 'anything_else':
    return 'user_party_rsvp_thanks'
  elif prev_wiz_act == 'anything_else' and next_wiz_act == 'goodbye_1':
    return 'user_party_rsvp_goodbye'
  elif prev_wiz_act == 'goodbye_1' and next_wiz_act == 'goodbye_1':
    return 'user_party_rsvp_goodbye'
  return None


def user_label_plane_book(task, graph, prev_wiz_act, next_wiz_act):
  del task, graph
  plane_find_acts = ('plane_flight_available', 'plane_flight_unavailable')
  plane_book_acts = ('plane_reservation_succeeded', 'plane_reservation_failed')
  if prev_wiz_act is None and next_wiz_act == 'hello':
    return 'user_hello'
  elif prev_wiz_act is None and is_ask_action(next_wiz_act):
    return 'user_plane_book_inform'
  elif prev_wiz_act == 'hello' and is_ask_action(next_wiz_act):
    return 'user_plane_book_inform'
  elif is_ask_action(prev_wiz_act):
    return 'user_plane_book_inform'
  elif next_wiz_act in plane_find_acts:
    return 'user_plane_book_inform'
  elif prev_wiz_act == 'plane_flight_available' and next_wiz_act in plane_book_acts:
    return 'user_plane_book_inform_book'
  elif prev_wiz_act == 'anything_else' and next_wiz_act == 'goodbye_2':
    return 'user_plane_book_goodbye'  # should be no more?
  elif prev_wiz_act in plane_book_acts and next_wiz_act == 'anything_else':
    return 'user_plane_book_thanks'
  elif prev_wiz_act in plane_book_acts and next_wiz_act == 'goodbye_2':
    return 'user_plane_book_goodbye'
  elif prev_wiz_act == 'plane_flight_unavailable' and next_wiz_act == 'anything_else':
    return 'user_plane_book_thanks'
  elif next_wiz_act == 'goodbye_2':
    return 'user_plane_book_goodbye'
  return None


def user_label_restaurant_book(task, graph, prev_wiz_act, next_wiz_act):
  del task, graph
  inform_booking_actions = ('restaurant_inform_booking_successful',
                            'restaurant_inform_booking_failed')
  if prev_wiz_act == 'anything_else' and next_wiz_act == 'goodbye_1':
    return 'user_restaurant_book_goodbye'
  elif prev_wiz_act == 'restaurant_inform_unavailable':
    if next_wiz_act == 'anything_else':
      return 'user_restaurant_book_thanks'
    else:
      # User needs to update slot values
      return 'user_restaurant_book_inform'
  elif (prev_wiz_act == 'restaurant_ask_confirm_booking' and
        next_wiz_act in inform_booking_actions):
    return 'user_restaurant_book_inform_book'
  elif (prev_wiz_act in inform_booking_actions and
        next_wiz_act in ENDING_ACTIONS):
    return 'user_restaurant_book_thanks'
  elif prev_wiz_act == 'restaurant_inform_booking_failed':
    if next_wiz_act in ENDING_ACTIONS:
      return 'user_restaurant_book_thanks'
    else:
      # User needs to update slot values
      return 'user_restaurant_book_inform'
  elif prev_wiz_act is None and next_wiz_act == 'hello':
    return 'user_hello'
  elif prev_wiz_act is None and is_ask_action(next_wiz_act):
    return 'user_restaurant_book_inform'
  elif prev_wiz_act == 'hello' and is_ask_action(next_wiz_act):
    return 'user_hello'
  elif is_ask_action(prev_wiz_act):
    return 'user_restaurant_book_inform'
  return None


def user_label_ride_book(task, graph, prev_wiz_act, next_wiz_act):
  del task, graph
  ending_actions = ENDING_ACTIONS + ('ride_bye',)
  found_ride_acts = ('ride_ask_confirm_booking', 'ride_inform_nothing_found')
  if prev_wiz_act is None and next_wiz_act == 'hello':
    return 'user_hello'
  elif prev_wiz_act is None and is_ask_action(next_wiz_act):
    return 'user_ride_book_inform'
  elif prev_wiz_act == 'hello' and is_ask_action(next_wiz_act):
    return 'user_ride_book_inform'
  elif is_ask_action(prev_wiz_act):
    return 'user_ride_book_inform'
  elif next_wiz_act in found_ride_acts:
    return 'user_ride_book_inform'
  elif prev_wiz_act == 'ride_inform_nothing_found' and next_wiz_act in found_ride_acts:
    return 'user_ride_book_inform'
  elif prev_wiz_act == 'ride_ask_confirm_booking' and next_wiz_act == 'ride_confirm_booking':
    return 'user_ride_book_confirm_booking'
  elif prev_wiz_act == 'ride_confirm_booking' and next_wiz_act in ending_actions:
    return 'user_ride_book_thanks'
  elif next_wiz_act == 'ride_bye':
    return 'user_ride_book_goodbye'
  return None


def user_label_ride_status(task, graph, prev_wiz_act, next_wiz_act):
  del task, graph
  if prev_wiz_act is None and next_wiz_act == 'hello':
    return 'user_hello'
  elif prev_wiz_act is None and is_ask_action(next_wiz_act):
    return 'user_ride_status_inform'
  elif prev_wiz_act == 'hello' and is_ask_action(next_wiz_act):
    return 'user_ride_status_inform'
  elif is_ask_action(prev_wiz_act):
    return 'user_ride_status_inform'
  elif next_wiz_act == 'ride_provide_booking_status':
    return 'user_ride_status_inform'
  elif prev_wiz_act == 'ride_provide_booking_status' and next_wiz_act == 'ride_bye':
    return 'user_ride_status_goodbye'
  elif prev_wiz_act == 'ride_provide_booking_status' and next_wiz_act == 'anything_else':
    return 'user_ride_status_thanks'
  elif prev_wiz_act == 'anything_else' and next_wiz_act == 'ride_bye':
    return 'user_ride_status_goodbye'
  return None


def user_label_spaceship_access_codes(task, graph, prev_wiz_act, next_wiz_act):
  del task, graph
  ending_actions = ENDING_ACTIONS + ('spaceship_bye',)
  if prev_wiz_act is None and next_wiz_act == 'hello':
    return 'user_hello'
  elif prev_wiz_act is None and is_ask_action(next_wiz_act):
    return 'user_spaceship_access_codes_inform'
  elif prev_wiz_act == 'hello' and is_ask_action(next_wiz_act):
    return 'user_spaceship_access_codes_inform'
  elif is_ask_action(prev_wiz_act):
    return 'user_spaceship_access_codes_inform'
  elif next_wiz_act == 'spaceship_inform_outcome':
    return 'user_spaceship_access_codes_inform'
  elif (prev_wiz_act == 'spaceship_inform_outcome' and
        next_wiz_act in ending_actions):
    return 'user_spaceshipcodes_thanks'
  elif is_ask_action(prev_wiz_act):
    return 'user_spaceship_access_codes_inform'
  elif next_wiz_act in ending_actions:
    return 'user_spaceshipcodes_goodbye'
  return None


def user_label_spaceship_life_support(task, graph, prev_wiz_act, next_wiz_act):
  del task, graph
  ending_actions = ENDING_ACTIONS + ('spaceship_bye',)
  if prev_wiz_act == 'anything_else' and next_wiz_act in ending_actions:
    return 'user_spaceship_life_support_goodbye'
  elif (prev_wiz_act == 'spaceship_inform_outcome' and
        next_wiz_act in ending_actions):
    return 'user_spaceship_life_support_thanks'
  elif prev_wiz_act in ending_actions and next_wiz_act in ending_actions:
    return 'user_spaceship_life_support_goodbye'
  elif is_ask_action(prev_wiz_act):
    return 'user_spaceship_life_support_inform'
  elif next_wiz_act == 'spaceship_inform_outcome':
    return 'user_spaceship_life_support_inform'
  elif prev_wiz_act is None and next_wiz_act == 'hello':
    return 'user_hello'
  elif prev_wiz_act == 'hello' and is_ask_action(next_wiz_act):
    return 'user_hello'
  return None


def user_label_trivia(task, graph, prev_wiz_act, next_wiz_act):
  del task, graph
  after_answer_acts = ('trivia_inform_answer_correct_ask_next',
                       'trivia_inform_answer_incorrect_ask_next',
                       'trivia_inform_answer_2_ask_next')
  if prev_wiz_act is None and next_wiz_act == 'hello':
    return 'user_hello'
  elif prev_wiz_act is None and is_ask_action(next_wiz_act):
    return 'user_trivia_inform'
  elif prev_wiz_act == 'hello' and is_ask_action(next_wiz_act):
    return 'user_trivia_inform'
  elif is_ask_action(prev_wiz_act):
    return 'user_trivia_inform'
  # elif next_wiz_act == 'trivia_ask_question':
  #   return 'user_trivia_goodbye'
  elif prev_wiz_act == 'trivia_ask_question' and next_wiz_act in after_answer_acts:
    return 'user_trivia_answer'
  elif prev_wiz_act in after_answer_acts and next_wiz_act == 'trivia_ask_question':
    return 'user_trivia_next_question'
  elif prev_wiz_act in after_answer_acts and next_wiz_act == 'anything_else':
    return 'user_trivia_thanks'
  elif prev_wiz_act in after_answer_acts and next_wiz_act == 'trivia_bye':
    return 'user_trivia_goodbye'
  elif next_wiz_act == 'trivia_bye':
    return 'user_trivia_goodbye'
  return None


# pylint: enable=missing-function-docstring

LABELING_FUNCTIONS = {
    'apartment_schedule': user_label_apartment_schedule,
    'apartment_search': user_label_catchall,
    'bank_balance': user_label_bank_balance,
    'bank_fraud_report': user_label_bank_fraud_report,
    'doctor_followup': user_label_doctor_followup,
    'doctor_schedule': user_label_doctor_schedule,
    'hotel_book': user_label_hotel_book,
    'hotel_search': user_label_catchall,
    'hotel_service_request': user_label_catchall,
    'meeting_schedule': user_label_meeting_schedule,
    'party_plan': user_label_party_plan,
    'party_rsvp': user_label_party_rsvp,
    'plane_book': user_label_plane_book,
    'plane_search': user_label_catchall,
    'restaurant_book': user_label_restaurant_book,
    'restaurant_search': user_label_catchall,
    'ride_book': user_label_ride_book,
    'ride_change': user_label_catchall,
    'ride_status': user_label_ride_status,
    'spaceship_access_codes': user_label_spaceship_access_codes,
    'spaceship_life_support': user_label_spaceship_life_support,
    'trivia': user_label_trivia,
    'trip_directions': user_label_trip_directions,
    'weather': user_label_catchall,
}


@dataclasses.dataclass
class StarApiParam:
  name: str
  readable_name: str
  type: str
  categories: list[str]
  required: bool


class StarApi:
  """Wrapper for STAR API JSON."""

  def __init__(self, api_json: Json):
    self._api_json = api_json

  @functools.cached_property
  def params(self) -> list[StarApiParam]:
    """Cache the params of this API."""
    params = []
    for param in self._api_json['input']:
      if param['Name'] == 'RequestType':
        continue
      params.append(
          StarApiParam(param['Name'], param['ReadableName'], param['Type'],
                       param.get('Categories'), param['Name']
                       in self._api_json['required']))
    return params

  @functools.cached_property
  def params_by_name(self) -> dict[str, StarApiParam]:
    """Build a dict to look up parameters by name."""
    params = {}
    for param in self._api_json['input']:
      if param['Name'] == 'RequestType':
        continue
      params[param['Name']] = StarApiParam(
          param['Name'],
          param['ReadableName'],
          param['Type'],
          param.get('Categories'),
          param['Name'] in self._api_json['required'],
      )
    return params


T = TypeVar('T')


# TODO(jeffreyzhao): Maybe need better type annotations?
class Ordering(Generic[T]):
  """Provides an ordering for a list of entities."""

  def __init__(self, entities: list[T], shuffle_idxs: bool = True):
    self._idx_to_entity = copy.deepcopy(entities)
    if shuffle_idxs:
      random.shuffle(self._idx_to_entity)
    self._entity_to_idx = {e: i for i, e in enumerate(self._idx_to_entity)}

  @property
  def idx_to_entity(self) -> list[T]:
    return self._idx_to_entity

  def __contains__(self, e: T) -> bool:
    return e in self._idx_to_entity

  def get_idx(self, e: T) -> int:
    return self._entity_to_idx[e]

  def get_entity(self, idx: int) -> T:
    return self._idx_to_entity[idx]

  def tolist(self) -> list[T]:
    return self._idx_to_entity

  def __iter__(self) -> Iterator[tuple[int, T]]:
    return enumerate(self._idx_to_entity)


def bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


@dataclasses.dataclass
class Example:
  """Example dataclass."""
  src: str
  tgt: str
  dialog_id: int
  turn: int
  metadata: Json
  policy_table: list[Json]

  def build_tf_example(self) -> tf.train.Example:
    """Converts this Example dataclass into a tf.Example."""
    policy_table_json_str = json.dumps(self.policy_table)

    return tf.train.Example(
        features=tf.train.Features(
            feature={
                'input': bytes_feature(self.src.encode('utf-8')),
                'value': bytes_feature(self.tgt.encode('utf-8')),
                'dialog_id': bytes_feature(str(self.dialog_id).encode('utf-8')),
                'turn': int64_feature(self.turn),
                'metadata': bytes_feature(
                    json.dumps(self.metadata).encode('utf-8')
                ),
                'policy_table': bytes_feature(
                    policy_table_json_str.encode('utf-8')
                ),
            }
        )
    )


@dataclasses.dataclass
class Options:
  """A dataclass to store configurations for data generation."""
  example_format: ExampleFormat
  shuffle_entity_indices: bool
  skip_custom_wizard_actions: bool


@dataclasses.dataclass
class StarGraph:
  """Wrapper for STAR task graph JSON."""
  json: Json
  replies: dict[str, str]
  slot_actions: dict[str, list[str]]
  graph: dict[str, str]
  r_graph: dict[str, str]

  @functools.cached_property
  def graph_reverse(self) -> dict[str, list[str]]:
    ret = collections.defaultdict(list)
    for ua, sa in self.graph.items():
      ret[sa].append(ua)
    return ret

  @functools.cached_property
  def intent_act(self) -> str:
    return list(self.json['replies'].keys())[2]

  @functools.cached_property
  def user_bye_act(self) -> str:
    """Find the user bye act."""
    bye_acts = []
    for act in self.replies:
      if act.startswith('user_') and 'bye' in act:
        bye_acts.append(act)
    assert len(bye_acts) == 1
    return bye_acts[0]

  @functools.cached_property
  def wiz_bye_act(self) -> str:
    """Find the system bye act."""
    bye_acts = []
    for act in self.replies:
      if not act.startswith('user_') and 'bye' in act:
        bye_acts.append(act)
    assert len(bye_acts) == 1
    return bye_acts[0]

  @functools.cached_property
  def ask_act_to_slot_name(self) -> dict[str, str]:
    ret = {}
    for slot, ask_acts in self.slot_actions.items():
      assert len(ask_acts) == 1
      ret[ask_acts[0]] = slot
    return ret

  @functools.cached_property
  def slot_to_user_inform_act(self) -> dict[str, list[str]]:
    r_graph_reverse = collections.defaultdict(list)
    for user_act, wiz_act in self.r_graph.items():
      r_graph_reverse[wiz_act].append(user_act)
    ret = {}
    for slot, ask_acts in self.slot_actions.items():
      assert len(ask_acts) == 1
      ret[slot] = r_graph_reverse[ask_acts[0]]
    return ret

  @functools.cached_property
  def all_user_inform_acts(self) -> set[str]:
    ret = set()
    for _, inform_useracts in self.slot_to_user_inform_act.items():
      for ua in inform_useracts:
        ret.add(ua)
    return ret

  @functools.cached_property
  def user_inform_act_to_slot(self) -> dict[str, str]:
    ret = {}
    for slot, acts in self.slot_to_user_inform_act.items():
      for act in acts:
        if 'forgot' not in act:
          ret[act] = slot
    return ret

  @functools.cached_property
  def slot_order(self) -> list[str]:
    return list(self.slot_actions.keys())


def entity_to_ind(entity: ...,
                  prefix_and_ords: Sequence[tuple[str, Ordering]]) -> str:
  for prefix, ord_ in prefix_and_ords:
    if entity in ord_:
      return prefix + str(ord_.get_idx(entity))
  raise ValueError('couldn\'t translate %s' % entity)


def ind_to_entity(ind: str, prefix_and_ords: Sequence[tuple[str,
                                                            Ordering]]) -> str:
  for prefix, ord_ in prefix_and_ords:
    if prefix == ind[0]:
      idx = int(ind[1:])
      return ord_.get_entity(idx)
  raise ValueError('couldn\'t translate %s' % ind)


def normal_policy(
    bs: BeliefState,
    act_hist: ActionHistory,
    task: str,
    api: StarApi,
    graph: StarGraph,
    convo_hist: list[str],
    primary_item: Json,
) -> list[str]:
  """Policy for normal STARv2 tasks."""
  del convo_hist, primary_item, task
  ret = []
  inform_useracts = graph.all_user_inform_acts

  if act_hist:
    for last_useract in act_hist[-1]:
      if last_useract == USER_CUSTOM_LABEL:
        ret.append(OUT_OF_SCOPE_LABEL)

      if last_useract == graph.user_bye_act:
        ret.append(graph.wiz_bye_act)

      if last_useract not in inform_useracts | set([graph.intent_act]):
        if last_useract in graph.graph:
          ret.append(graph.graph[last_useract])

  if 'anything_else' in ret:
    ret.append(graph.wiz_bye_act)

  # all required slots -> query
  query_label = 'query' if 'query' in graph.replies else 'query_check'
  if all(p.name in bs for p in api.params if p.required):
    ret.append(query_label)

  # for p in api.params:
  for slot in graph.slot_order:
    p = api.params_by_name[slot]
    if p.name not in bs:
      if p.name not in graph.slot_actions:
        assert p.name in ('TripLengthMinutes', 'Price')
        # print(f'SKIP {p.name}')
        continue
      ask_sysact = graph.slot_actions[p.name]
      assert len(ask_sysact) == 1
      ask_sysact = ask_sysact[0]
      ret.append(ask_sysact)

  return ret


def trip_policy(
    bs: BeliefState,
    act_hist: ActionHistory,
    task: str,
    api: StarApi,
    graph: StarGraph,
    convo_hist: list[str],
    primary_item: Json,
) -> list[str]:
  """Policy for STARv2 trip_direction task."""
  if act_hist and len(convo_hist) >= 3:
    for last_useract in act_hist[-1]:
      if last_useract in (
          'user_trip_directions_continue',
          'user_trip_directions_ok',
      ):
        last_sys_utt = convo_hist[-3]
        travel_mode = primary_item.get('TravelMode', '')
        # See some travel modes to be all lowercase, so make the first char
        # upper case
        instrs = primary_item.get(f'{travel_mode}Instructions', None)
        detailed_instrs = primary_item.get(
            f'Detailed{travel_mode}Instructions', None
        )
        if travel_mode and instrs and detailed_instrs:
          if detailed_instrs[-1] in last_sys_utt:
            return ['trip_instructions_done']
          elif instrs[-1] in last_sys_utt:
            return ['trip_instructions_done']
          elif detailed_instrs[-2] in last_sys_utt:
            return ['trip_inform_last_step_and_done']
          elif instrs[-2] in last_sys_utt:
            return ['trip_inform_last_step_and_done']
  return normal_policy(bs, act_hist, task, api, graph, convo_hist, primary_item)


def trivia_policy(
    bs: BeliefState,
    act_hist: ActionHistory,
    task: str,
    api: StarApi,
    graph: StarGraph,
    convo_hist: list[str],
    primary_item: Json,
) -> list[str]:
  """Policy for STARv2 trivia task."""
  if act_hist and len(convo_hist) >= 2:
    last_user_utt = convo_hist[-2]
    for last_useract in act_hist[-1]:
      if last_useract == 'user_trivia_answer':
        answer = primary_item.get('Answer', None)
        if answer:
          if answer.lower() in last_user_utt.lower():
            return ['trivia_inform_answer_correct_ask_next']
          else:
            return ['trivia_inform_answer_incorrect_ask_next']
  return normal_policy(bs, act_hist, task, api, graph, convo_hist, primary_item)


def bank_policy(
    bs: BeliefState,
    act_hist: ActionHistory,
    task: str,
    api: StarApi,
    graph: StarGraph,
    convo_hist: list[str],
    primary_item: Json,
) -> list[str]:
  """Policy for STARv2 bank_balance, bank_fraud_report tasks."""
  del api, convo_hist, primary_item
  ret = []

  forgot_acts = ['user_bank_forgot_account_number', 'user_bank_forgot_pin']
  inform_useracts = graph.all_user_inform_acts

  if act_hist:
    for last_useract in act_hist[-1]:
      if last_useract == USER_CUSTOM_LABEL:
        ret.append(OUT_OF_SCOPE_LABEL)

      if last_useract == graph.user_bye_act:
        ret.append(graph.wiz_bye_act)

      if last_useract not in inform_useracts | set([graph.intent_act]):
        if last_useract in graph.graph:
          ret.append(graph.graph[last_useract])

  if 'anything_else' in ret:
    ret.append(graph.wiz_bye_act)

  seen_useracts = set()
  for turn, turn_acts in enumerate(act_hist):
    if turn % 2 == 0:
      for act in turn_acts:
        seen_useracts.add(act)

  first_auth_slots = ['FullName', 'AccountNumber', 'PIN']
  second_auth_slots = [
      'FullName',
      'DateOfBirth',
      'SecurityAnswer1',
      'SecurityAnswer2',
  ]
  if task == 'bank_fraud_report':
    first_auth_slots.append('FraudReport')
    second_auth_slots.append('FraudReport')

  if all(slot in bs for slot in first_auth_slots):
    ret.append('query')
  if all(slot in bs for slot in second_auth_slots):
    ret.append('query')

  is_second_auth = any(fa in seen_useracts for fa in forgot_acts)
  slots = second_auth_slots if is_second_auth else first_auth_slots
  for slot in slots:
    if slot not in bs:
      slot_req_sysact = graph.slot_actions[slot]
      assert len(slot_req_sysact) == 1
      slot_req_sysact = slot_req_sysact[0]
      ret.append(slot_req_sysact)

  return ret


def policy_with_hist(
    bs: BeliefState,
    act_hist: ActionHistory,
    task: str,
    api: StarApi,
    graph: StarGraph,
    convo_hist: list[str],
    primary_item: Json,
) -> list[str]:
  """AnyTOD policy function."""
  if task in ('bank_balance', 'bank_fraud_report'):
    rec_sysacts = bank_policy(
        bs, act_hist, task, api, graph, convo_hist, primary_item
    )
  elif task == 'trivia':
    rec_sysacts = trivia_policy(
        bs, act_hist, task, api, graph, convo_hist, primary_item
    )
  elif task == 'trip_directions':
    rec_sysacts = trip_policy(
        bs, act_hist, task, api, graph, convo_hist, primary_item
    )
  else:
    rec_sysacts = normal_policy(
        bs, act_hist, task, api, graph, convo_hist, primary_item
    )

  # Deduplicate recommended system actions.
  seen_rec_sysacts = set()
  ret = []
  for sa in rec_sysacts:
    if sa not in seen_rec_sysacts:
      ret.append(sa)
      seen_rec_sysacts.add(sa)
  return ret


class StarDialog:
  """Class wrapping a STAR dialogue."""

  def __init__(self, dialog_json: Json, api: StarApi, graph: StarGraph,
               options: Options):
    self._json = dialog_json
    self._api = api
    self._graph = graph
    self._options = options

  @functools.cached_property
  def task(self) -> str:
    # assert len(self._json['Scenario']['WizardCapabilities']) == 1
    return self._json['Scenario']['WizardCapabilities'][0]['Task']

  @functools.cached_property
  def is_multitask(self) -> bool:
    return self._json['Scenario']['MultiTask']  # pytype: disable=bad-return-type

  @functools.cached_property
  def is_happy(self) -> bool:
    return self._json['Scenario']['Happy']  # pytype: disable=bad-return-type

  @property
  def json(self) -> Json:
    return self._json

  @property
  def dialog_id(self) -> str:
    return self._json['DialogueID']  # pytype: disable=bad-return-type

  @property
  def events(self) -> Json:
    return self._json['Events']

  def get_event(self, turn: int) -> Json:
    return self.events[turn]

  def get_closest_event(
      self,
      turn: int,
      criteria: Callable[[Json], bool],
      reverse: bool = False) -> tuple[Optional[int], Optional[Json]]:
    """Gets the cloesst event from some turn that satisfies some criteria."""
    if not reverse:
      turn_range = range(turn + 1, len(self.events))
    else:
      turn_range = range(turn - 1, -1, -1)

    for closest_turn in turn_range:
      event = self.get_event(closest_turn)
      if criteria(event):
        return closest_turn, event
    return None, None

  def build_user_labels(self) -> tuple[UserLabels, list[int]]:
    """Builds user labels according to user action labeling functions."""
    user_labels = {}
    unlabeled_user_turns = []
    for turn, event in enumerate(self.events):
      if is_user_event(event):
        # Get most recent previous wizard turn.
        _, prev_sys_event = self.get_closest_event(
            turn, is_wizard_event, reverse=True
        )
        # Get next upcoming wizard turn.
        _, next_sys_event = self.get_closest_event(turn, is_wizard_event)

        prev_wiz_act = (
            get_action_label(prev_sys_event)
            if prev_sys_event is not None
            else None
        )
        next_wiz_act = (
            get_action_label(next_sys_event)
            if next_sys_event is not None
            else None
        )

        user_label = LABELING_FUNCTIONS[self.task](
            self.task, self._graph, prev_wiz_act, next_wiz_act
        )

        if next_wiz_act == OUT_OF_SCOPE_LABEL:
          user_label = USER_CUSTOM_LABEL
        if user_label is None:
          user_label = USER_BAD_LABEL
          unlabeled_user_turns.append(turn)
        user_labels[turn] = user_label
      elif is_result_event(event):
        _, next_wiz_event = self.get_closest_event(turn, is_wizard_event)
        # Find the db_* actions that precede the next wizard action.
        # TODO(jeffreyzhao): Multiple db_* actions can precede here! e.g.
        # trip_directions.
        if next_wiz_event:
          next_wiz_act = get_action_label(next_wiz_event)
          db_acts_from_graph = [
              k
              for k, v in self._graph.graph.items()
              if v == next_wiz_act and k.startswith('db_')
          ]
          if next_wiz_act == WIZARD_CUSTOM_LABEL:
            continue
          if not db_acts_from_graph:
            # Probably an unexpected wizard action.
            print('trouble inferring db acts! next_wiz_act %s', next_wiz_act)
            user_labels[turn] = RESULT_BAD_LABEL
          if db_acts_from_graph:
            user_labels[turn] = self.get_db_act_label(event, db_acts_from_graph)
    return user_labels, unlabeled_user_turns

  def __str__(self):
    lines = [f'{self.dialog_id}']
    for turn, event in enumerate(self.events):
      if is_user_event(event) or is_wizard_event(event):
        text = event['Text']
        speaker = get_event_speaker(event)
        lines.append(f'{turn} {speaker}: {text}')
    return '\n'.join(lines)

  # TODO(jeffreyzhao): prepend is hack!!!
  def build_event_str(
      self, event: Event, param_name_ordering: Ordering, prepend=''
  ) -> str:
    """Generates an event string."""
    if is_user_event(event) or is_wizard_event(event):
      text = event['Text']
      speaker = get_event_speaker(event)
      speaker_token = USER_MARKER if speaker == 'user' else SYSTEM_MARKER
      return f'{speaker_token} {prepend}{text}'
    elif is_query_event(event):
      # Make call with params ordered by index.
      arg_ind_to_value = [
          (param_name_ordering.get_idx(k), v)
          for k, v in get_query_event_belief_state(event).items()
      ]
      arg_ind_to_value.sort()
      arg_ind_to_value_str = ' '.join(
          [f'p{ind}={val}' for ind, val in arg_ind_to_value])
      return f'{API_CALL_MARKER} {arg_ind_to_value_str}'
    elif is_result_event(event):
      if 'Item' not in event:
        # TODO(jeffreyzhao): Why does this occur?
        item = ''
      else:
        item = ' '.join(f'{k}={v}' for k, v in event['Item'].items())
      return f'{API_RETURN_MARKER} {item}'
    else:
      raise ValueError(event)

  def get_action_desc(self, act: str) -> str:
    """Gets the action description for some action."""
    if act == WIZARD_CUSTOM_LABEL:
      return 'system performing custom action'
    elif act == USER_CUSTOM_LABEL:
      return 'user performing custom action'
    elif act == f'user_{self.task}_inform':
      return 'user is informing the agent of slot values'
    elif act == QUERY_LABEL:
      return 'system making query'
    elif act == RESULT_LABEL:
      return 'query result'
    elif act == 'out_of_scope':
      # TODO(jeffreyzhao): out_of_scope for all tasks
      return f'I am sorry, I don\'t quite understand what you mean. I am only able to help you with {self.task}.'
    elif act == 'anything_else':
      return 'Is there anything else that I can do for you?'
    elif act == RESULT_CUSTOM_LABEL:
      return 'unknown query result'
    elif act in self._graph.user_inform_act_to_slot:
      slot = self._graph.user_inform_act_to_slot[act]
      return f'user is informing {slot}'
    else:
      return self._graph.replies[act].replace('[', '{').replace(']', '}')

  def build_actions_anytod_context_str(
      self,
      sys_node_ordering: Ordering,
      user_node_ordering: Ordering,
      param_name_ordering: Ordering,
  ) -> str:
    """Builds a context string for the schema action graph."""

    param_pieces = []
    sysact_pieces = []
    useract_pieces = []
    readable_param_names = {p.name: p.readable_name for p in self._api.params}
    for idx, param in param_name_ordering:
      param_pieces.append(f'p{idx}={readable_param_names[param]}')
    for idx, sys_node in sys_node_ordering:
      sysact_pieces.append(f's{idx}={self.get_action_desc(sys_node)}')
    for idx, user_node in user_node_ordering:
      useract_pieces.append(f'u{idx}={self.get_action_desc(user_node)}')
    param_str = '; '.join(param_pieces)
    sysact_str = '; '.join(sysact_pieces)
    useract_str = '; '.join(useract_pieces)
    return (
        f'[ params ] {param_str} [ user actions ] {useract_str} [ system'
        f' actions ] {sysact_str}'
    )

  def build_transitions_anytod_context_str(
      self,
      sys_node_ordering: Ordering,
      user_node_ordering: Ordering,
      trans_ord: Ordering,
      param_name_ordering: Ordering,
  ) -> str:
    """Builds a context string for the schema action graph."""

    pieces = []
    readable_param_names = {p.name: p.readable_name for p in self._api.params}
    for idx, param in param_name_ordering:
      pieces.append(f'p{idx}={readable_param_names[param]}')
    for idx, sys_node in sys_node_ordering:
      pieces.append(f's{idx}={self.get_action_desc(sys_node)}')
    for idx, (wiz_node, user_node) in trans_ord:
      if wiz_node is None:
        continue
      wiz_act_desc = self.get_action_desc(wiz_node)
      user_act_desc = self.get_action_desc(user_node)
      pieces.append(
          f't{idx}={SYSTEM_MARKER} {wiz_act_desc} {USER_MARKER} {user_act_desc}'
      )
    for idx, user_node in user_node_ordering:
      pieces.append(f'u{idx}={self.get_action_desc(user_node)}')
    return ' '.join(pieces)

  def build_params_context_str(self, param_name_ordering: Ordering) -> str:
    """Builds a context string for the possible API parameters."""
    readable_param_names = {p.name: p.readable_name for p in self._api.params}
    context_str_pieces = []
    for idx, param_name in param_name_ordering:
      context_str_pieces.append(
          f'p{idx} = "{readable_param_names[param_name]}"')
    return ' '.join(context_str_pieces)

  def get_all_transitions(self) -> collections.Counter[Trans]:
    """Gets all wizard to user transitions from actual data."""
    # TODO(jeffreyzhao): Maybe we want to read these from graph JSON.
    # From some empirical tests this doesn't matter.
    all_trans = collections.Counter()
    for turn, event in enumerate(self.events):
      if is_user_or_result_event(event):
        user_act = get_action_label_from_json(event)
        _, prev_wiz_event = self.get_closest_event(
            turn, is_wizard_or_query_event, reverse=True
        )
        prev_wiz_act = (
            get_action_label_from_json(prev_wiz_event)
            if prev_wiz_event
            else None
        )

        if prev_wiz_act == WIZARD_CUSTOM_LABEL or user_act == USER_CUSTOM_LABEL:
          continue
        all_trans[(prev_wiz_act, user_act)] += 1
    return all_trans  # pytype: disable=bad-return-type

  def build_updated_json(self, user_labels: dict[int, str],
                         dst_preds: dict[str, int]) -> Json:
    """Given DST results and user action labels, update annotations in JSON."""
    new_json = copy.deepcopy(self._json)
    new_json['PredictedSlotActivationTurns'] = dst_preds

    # T5 converts truecased slot names to lowercase
    # Translate them back
    slot_name_true_case = {p.name.lower(): p.name for p in self._api.params}
    dst_preds = {
        slot_name_true_case[slot.lower()]: turn
        for slot, turn in dst_preds.items()
    }

    # List of active slots at each turn
    turn_to_active_slots = collections.defaultdict(list)
    for slot, active_turn in dst_preds.items():
      if slot in self._graph.slot_to_user_inform_act:
        turn_to_active_slots[active_turn].append(slot)
      else:
        print('slot %s s active but not known in graph', slot)

    # TODO(jeffreyzhao): This belief state annotation is not very good.
    def get_future_ground_truth_belief_state(turn):
      _, query_event = self.get_closest_event(turn, is_query_event)
      gt_belief_state = (
          get_query_event_belief_state(query_event) if query_event else {}
      )
      if 'RequestType' in gt_belief_state:
        del gt_belief_state['RequestType']
      # if self.task in ('bank_fraud_report', 'bank_balance'):
      #   if 'AccountNumber' not in gt_belief_state:
      #     gt_belief_state['AccountNumber'] = 'forgot'
      #   if 'PIN' not in gt_belief_state:
      #     gt_belief_state['PIN'] = 'forgot'
      return gt_belief_state

    def get_inform_and_forgot_acts(slot_acts):
      if len(slot_acts) != 1:
        assert self.task in ('bank_balance', 'bank_fraud_report')
      forgot_slot_act = None
      inform_slot_act = None
      for act in self._graph.slot_to_user_inform_act[slot]:
        if 'forgot' in act:
          forgot_slot_act = act
        else:
          inform_slot_act = act
      return inform_slot_act, forgot_slot_act

    partial_belief_state = {}
    for turn, event in enumerate(new_json['Events']):
      gt_belief_state = get_future_ground_truth_belief_state(turn)
      # TODO(jeffreyzhao): Slot values might change in future!!
      slot_action_labels = []

      if is_user_or_result_event(event):
        if dst_preds:
          for slot in turn_to_active_slots[turn]:
            if slot in gt_belief_state:
              partial_belief_state[slot] = gt_belief_state[slot]
              inform_act, _ = get_inform_and_forgot_acts(
                  self._graph.slot_to_user_inform_act[slot]
              )
              slot_action_labels.append(inform_act)
            else:
              _, forgot_act = get_inform_and_forgot_acts(
                  self._graph.slot_to_user_inform_act[slot]
              )
              slot_action_labels.append(forgot_act)
          event['PredictedBeliefState'] = copy.deepcopy(partial_belief_state)  # pytype: disable=unsupported-operands
        else:
          event['PredictedBeliefState'] = {}  # pytype: disable=unsupported-operands

      if is_wizard_event(event):
        if 'ActionLabel' in event:
          event['ActionLabel'] = [event['ActionLabel']]  # pytype: disable=unsupported-operands
      elif is_query_event(event):
        event['ActionLabel'] = [get_query_event_act(event)]  # pytype: disable=unsupported-operands
      elif is_user_event(event):
        assert turn in user_labels
        if user_labels[turn] == user_inform_act(self.task):
          event['ActionLabel'] = slot_action_labels  # pytype: disable=unsupported-operands
        else:
          event['ActionLabel'] = [user_labels[turn]] + slot_action_labels  # pytype: disable=unsupported-operands

        if not event['ActionLabel']:
          event['ActionLabel'] = USER_CUSTOM_LABEL  # pytype: disable=unsupported-operands
      elif is_result_event(event):
        _, next_wiz_event = self.get_closest_event(turn, is_wizard_event)
        # Find the db_* actions that precede the next wizard action.
        # TODO(jeffreyzhao): Multiple db_* actions can precede here! e.g.
        # trip_directions.
        if next_wiz_event:
          next_wiz_act = get_action_label(next_wiz_event)
          db_acts_from_graph = [
              k
              for k, v in self._graph.graph.items()
              if v == next_wiz_act and k.startswith('db_')
          ]
          if not db_acts_from_graph:
            event['ActionLabel'] = []  # pytype: disable=unsupported-operands
          else:
            event['ActionLabel'] = self.get_db_act_label(  # pytype: disable=unsupported-operands
                event, db_acts_from_graph
            )
    return new_json

  def get_db_act_label(self, event: Event, poss_db_acts: list[str]) -> str:
    """Infer db_.* action label."""
    if len(poss_db_acts) == 1:
      return poss_db_acts[0]
    for poss_db_act in poss_db_acts:
      poss_db_act_desc = self._graph.replies[poss_db_act]
      poss_db_act_items = [
          _.strip().split('=')
          for _ in poss_db_act_desc.strip().split(';')
          if _.strip()
      ]
      poss_db_act_items = {k.strip(): v.strip() for k, v in poss_db_act_items}
      if self.task == 'trip_directions':
        field = 'TravelMode'
      else:
        field = 'Message'
      if poss_db_act_items[field] == event['Item'][field]:
        return poss_db_act
    raise ValueError('Could not infer database action for %s' % event)


@dataclasses.dataclass
class StarData:
  dialogs: dict[DialogId, StarDialog]
  task_to_ids: dict[str, list[DialogId]]
  task_to_api: dict[str, StarApi]
  task_to_graph: dict[str, StarGraph]
  all_ask_acts: set[str]


def read_file(filename: str) -> str:
  with tf.io.gfile.GFile(filename) as f:
    return f.read()


def load_star_api_jsons(data_dir: str) -> dict[str, StarApi]:
  """Loads all STAR API JSONs."""
  task_to_api = {}
  for filename in tf.io.gfile.glob(os.path.join(data_dir, 'apis', 'apis', '*.json')):
    task = os.path.splitext(os.path.basename(filename))[0]
    with tf.io.gfile.GFile(filename) as f:
      task_to_api[task] = StarApi(json.load(f, object_hook=Json))
  return task_to_api


def load_star_graph_jsons(data_dir: str) -> dict[str, StarGraph]:
  """Loads all STAR task graph JSONs."""
  task_to_graph = {}
  for task_dir in tf.io.gfile.glob(os.path.join(data_dir, 'tasks', '*')):
    task = os.path.basename(task_dir)
    with tf.io.gfile.GFile(os.path.join(task_dir, task + '.json')) as f:
      graph_json = json.load(f)
      if STAR_VERSION == StarVersion.V2 and 'slot_actions' not in graph_json:
        raise ValueError(f'{task} has no slot_actions!')
      task_to_graph[task] = StarGraph(
          json=graph_json,
          replies=graph_json['replies'],
          slot_actions=graph_json.get('slot_actions', {}),
          graph=graph_json['graph'],
          r_graph=graph_json['r_graph'],
      )
  return task_to_graph


def load_star_jsons(data_dir: str, options: Options) -> StarData:
  """Loads all STAR dataset JSONs."""
  filenames = tf.io.gfile.glob(os.path.join(data_dir, 'dialogues', '*'))

  # TODO(jeffreyzhao): Parallelize below.
  start_time = time.time()
  dialog_json_strs = [read_file(f) for f in filenames]
  print(
      f'Reading dialogues_*.json files took {time.time() - start_time} seconds.'
  )

  task_to_api = load_star_api_jsons(data_dir)
  task_to_graph = load_star_graph_jsons(data_dir)

  all_ask_acts = set()
  for _, graph in task_to_graph.items():
    for _, actions in graph.slot_actions.items():
      assert len(actions) == 1
      all_ask_acts.add(actions[0])

  # TODO(jeffreyzhao): Avoid this global
  global ALL_ASK_ACTS
  ALL_ASK_ACTS = all_ask_acts

  dialogs = {}
  for js_str in dialog_json_strs:
    js = json.loads(js_str, object_hook=Json)
    # TODO(jeffreyzhao): Multitask
    task = js['Scenario']['WizardCapabilities'][0]['Task']
    dialog = StarDialog(js, task_to_api[task], task_to_graph[task], options)
    if not dialog.is_multitask:
      dialogs[js['DialogueID']] = dialog

  task_to_ids = collections.defaultdict(list)
  for dialog_id, dialog in dialogs.items():
    task_to_ids[dialog.task].append(dialog_id)

  return StarData(dialogs, task_to_ids, task_to_api, task_to_graph,
                  all_ask_acts)
