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

"""Tests for create_star_data_for_d3st_labeling."""
import os
import random
from unittest import mock

from absl import flags
from absl.testing import parameterized
from task_oriented_dialogue.end2end.anytod import create_star_data_for_d3st_labeling
from task_oriented_dialogue.end2end.anytod import starv2_lib
import tensorflow as tf

FLAGS = flags.FLAGS

TEST_DIR = 'zero_shot_task_oriented_dialog/testdata'


class CreateStarDataForD3stLabelingTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    random.seed(123)
    self._mock = mock.patch.object(
        create_star_data_for_d3st_labeling,
        'TASK_DESCS',
        {'apartment_schedule': 'user wants to schedule apartment viewing'},
    )
    self._test_dir = os.path.join(FLAGS.test_srcdir, TEST_DIR)

  def test_basic(self):
    options = starv2_lib.Options(
        starv2_lib.ExampleFormat.TRANSITIONS_ANYTOD, False, False
    )
    starv2_lib.set_star_version(starv2_lib.StarVersion.V1)
    data = starv2_lib.load_star_jsons(
        os.path.join(self._test_dir, 'star_data'), options
    )
    with self._mock:
      task_examples = (
          create_star_data_for_d3st_labeling.generate_star_d3st_examples(data)
      )
    self.assertEqual(
        task_examples['apartment_schedule'][3].inp,
        (
            '0=Housing Company 0a) One on Center Apartments 0b) Shadyside'
            ' Apartments 0c) North Hill Apartments 1=Day 1a) Wednesday 1b)'
            ' Friday 1c) Saturday 1d) Monday 1e) Sunday 1f) Thursday 1g)'
            ' Tuesday 2=Start Time Hour 2a) 8 am 2b) 9 am 2c) 10 am 2d) 11 am'
            ' 3=Message 4=Application Fee Paid 4a) No 4b) Yes 5=Renter Name'
            ' i0=user wants to schedule apartment viewing [user] Hello [system]'
            ' Hello, how can I help? [user] I would like to schedule a viewing'
            ' at North HIll Apartments. [system] Could you give me your name,'
            ' please? [user] Ben [system] What day would you like to make the'
            " booking for? [user] Let's make it Saturday at 9am"
        ),
    )


if __name__ == '__main__':
  tf.test.main()
