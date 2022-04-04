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

"""The dialogue state tracking module."""

from typing import Dict, List

from common import dialog_pb2
from common import schema
from common import system_summary


class AbstractDST:
  """Abstract class for a schema guided dialogue state tracker."""

  def update_service_state(self, partial_dialog: dialog_pb2.Dialog,
                           summary: system_summary.SystemSummary):
    """Adds the updated dialogue state for the service in the last turn.

    The last turn of the partial dialogue must be a user utterance. The returned
    state contains the summary of the dialogue history.

    Args:
      partial_dialog: A Dialog proto containing turns in the dialogue history.
      summary: The summary object corresponding to the service.
    """
    del partial_dialog, summary  # Unused.
    pass

  def get_transfer_candidates(
      self, partial_dialog: dialog_pb2.Dialog,
      service_schema: schema.ServiceSchema
  ) -> Dict[str, List[dialog_pb2.AttrValue]]:
    """Identifies the slot values that may be transferred between services.

    Args:
      partial_dialog: A Dialog proto containing the history utterances with
        previously predicted dialogue state annotations for all services.
      service_schema: The schema of the service for which slot transfer
        candidates will be generated.

    Returns:
      A dict mapping slots of the provided schema to candidate values obtained
      from the dialogue state of other services in the history.
    """
    # The base class provides a logic which uses fields in the schema.
    del partial_dialog, service_schema  # Unused.
    return {}
