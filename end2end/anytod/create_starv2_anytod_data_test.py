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

"""Tests for create_starv2_anytod_data."""
import os
import random
from unittest import mock

from absl import flags
from absl.testing import parameterized
from task_oriented_dialogue.end2end.anytod import create_starv2_anytod_data
from task_oriented_dialogue.end2end.anytod import starv2_lib
import tensorflow as tf

FLAGS = flags.FLAGS

TEST_DIR = "zero_shot_task_oriented_dialog/testdata"


class CreateStarv2AnytodDataTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    random.seed(123)
    self._mock = mock.patch.object(
        create_starv2_anytod_data,
        "TASKS",
        ["apartment_schedule"],
    )
    self._test_dir = os.path.join(FLAGS.test_srcdir, TEST_DIR)

  def test_basic(self):
    options = starv2_lib.Options(
        starv2_lib.ExampleFormat.TRANSITIONS_ANYTOD, False, False
    )
    starv2_lib.set_star_version(starv2_lib.StarVersion.V1)
    data = starv2_lib.load_star_jsons(
        os.path.join(self._test_dir, "starv2_data"), options
    )
    with self._mock:
      task_examples = create_starv2_anytod_data.generate_examples(data)

    inp = (
        "[ params ] p0=Application Fee Paid a) no b) yes; p1=Housing Company"
        " a) shadyside apartments b) one on center apartments c) north hill "
        "apartments; p2=Message; p3=Renter Name; p4=Day a) tuesday b) monday"
        " c) thursday d) friday e) saturday f) wednesday g) sunday; p5=Start"
        " Time Hour [ user actions ] u0=query result: apartment viewing time"
        " slot is available; u1=user is saying hello; u2=user is informing "
        "p0; u3=user is informing p5; u4=user doesn't want to book this "
        "apartment viewing anymore; u5=user is saying thanks; u6=user wants "
        "to schedule an apartment viewing; u7=user is informing p3; u8=query"
        " result: apartment viewing was scheduled; u9=query result: "
        "apartment viewing time is unavailable; u10=user is informing p1; "
        "u11=user is doing something out of scope; u12=user is informing p2;"
        " u13=user is informing p4; u14=user is confirming apartment viewing"
        " [ system actions ] s0=ask user if they need anything else; "
        "s1=request p3 from the user; s2=request p1 from the user; s3=inform"
        " user the apartment viewing time slot is available; s4=inform user "
        "the apartment viewing time slot is unavailable; s5=say goodbye to "
        "user; s6=inform user the apartment viewing was booked successfully;"
        " s7=query to check apartment viewing is available; s8=request p0 "
        "from the user; s9=query to book apartment viewing; s10=request "
        "appointment end time from user; s11=tell user you don't understand "
        "what they want; s12=tell user you don't understand what they want; "
        "s13=request p4 from the user; s14=request p2 from the user; "
        "s15=request p5 from the user; s16=say hello to user [ conversation "
        "] [ user ] . Hello [ system ] Hello, how can I help? [ user ] I "
        "would like to schedule a viewing at North HIll Apartments. [ system"
        " ] Could you give me your name, please? [ user ] Ben"
    )
    val1 = (
        "[ belief state ] p1=c; p3=ben [ user actions ] u1; s16; u6 u10; s1; u7"
    )
    val2 = "[ recommended actions ] s13 s15 s8 s14"
    val3 = (
        "[ system actions ] s13 [ response ] [ system ] What day would you "
        "like to make the booking for?"
    )

    self.assertEqual(task_examples[(1005, "train")][4].inp, inp)
    self.assertEqual(task_examples[(1005, "train")][4].val, val1)
    self.assertEqual(
        task_examples[(1005, "train")][5].inp, " ".join([inp, val1, val2])
    )
    self.assertEqual(task_examples[(1005, "train")][5].val, val3)
    self.assertEqual(task_examples[(1005, "test")][2].inp, inp)
    self.assertEqual(
        task_examples[(1005, "test")][2].val, " ".join([val1, val2, val3])
    )

    self.assertLen(task_examples[(1005, "train")], 6)
    self.assertLen(task_examples[(1005, "test")], 11)


if __name__ == "__main__":
  tf.test.main()
