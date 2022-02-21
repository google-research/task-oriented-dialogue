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

"""Script to process SGD dialogs for finetuning T5."""

import csv
import json
import os
import random

from absl import app
from absl import flags
from generation import utterance_generator
import tensorflow as tf


flags.DEFINE_string("sgd_dir", None, "Directory containing the SGD dataset.")
flags.DEFINE_string("output_dir", None,
                    "Directory where datasets will be created.")
flags.DEFINE_boolean("delexicalize", False,
                     "Whether to delexicalize non-categorical slots")
flags.DEFINE_string("templates_dir", None,
                    "Directory contains utterance templates.")

FLAGS = flags.FLAGS

SEPARATOR = " | "
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FEWSHOT_IDS_DIR = f"{_CURRENT_DIR}/fewshot_splits"


def _remove_newline_and_tabs(s):
  return s.replace("\n", " ").replace("\t", " ")


class Preprocessor:
  """Utilities to prepare SGD dialogs into the text-to-text format."""

  def __init__(self, schemas_path, input_representation_type="naive"):
    """Inits a preprocessor object for preparing dataset.

    Args:
      schemas_path: Path to json file with schemas for all domains.
      input_representation_type: One of {'naive', 'schema', 't2g2'}. Controls
        how the structured data is represented in a textual form. 'naive' - This
        is the simplest form, representing the input as a series of slot value
        pairs. 'schema_guided' - Similar to naive, but adds the description of
        each slot. 't2g2' - Represents each slot value pair in the form of a
        simple template.
        Refer to https://arxiv.org/abs/2004.15006 for examples.
    """
    self.schemas_path = schemas_path
    self.slots_str_repr = {}
    self.service_str_repr = {}
    self.input_representation_type = input_representation_type
    self.add_slot_desc = (self.input_representation_type == "schema_guided")
    self.schemas_str_repr = self.preprocess_schemas(schemas_path)
    self._utterance_gen = utterance_generator.TemplateUtteranceGenerator(
        FLAGS.templates_dir)

  def preprocess_slot(self, slot):
    slot_str = f"name={slot['name']},description={slot['description']}"
    if slot["possible_values"]:
      slot_str = f"{slot_str},examples ={','.join(slot['possible_values'])}"
    return slot_str

  def load_schema(self, schemas_path, service):
    schemas = json.load(tf.io.gfile.GFile(schemas_path))
    for schema in schemas:
      if schema["service_name"] == service:
        return schema
    return schemas

  def preprocess_schemas(self, schemas_path):
    schemas = json.load(tf.io.gfile.GFile(schemas_path))
    schemas_str_repr = {
        schema["service_name"]: self.preprocess_schema(schema)
        for schema in schemas
    }
    return schemas_str_repr

  def get_domain_from_service(self, service):
    return service.split("_")[0]

  def preprocess_schema(self, schema):
    """Load service names, slots and their descriptions."""
    service_str = (f"service_name={schema['service_name']},"
                   "description={schema['description']}")
    self.service_str_repr[schema["service_name"]] = service_str
    schema_str_parts = [service_str]
    slots = schema["slots"]
    for slot in slots:
      slot_str = self.preprocess_slot(slot)
      schema_str_parts.append(slot_str)
      self.slots_str_repr[(schema["service_name"], slot["name"])] = slot_str
    schema_str = SEPARATOR.join(schema_str_parts).lower()
    return schema_str

  def preprocess_target_utterance(self, turn, schema=None):
    if FLAGS.delexicalize:
      return self._utterance_gen.get_delexicalized_utterance(turn, schema)
    else:
      return turn["utterance"]

  def preprocess_turn(self, turn, schema=None):
    """Convert a dialog turn into a textual representation."""
    if self.input_representation_type == "t2g2":
      robot_utterance = self._utterance_gen.get_robot_utterance(turn, schema)
      return turn["speaker"] + SEPARATOR + robot_utterance
    turn_str_parts = [turn["speaker"]]
    for frame in turn["frames"]:
      turn_str_parts.append(self.preprocess_frame(frame))
    return SEPARATOR.join(turn_str_parts)

  def preprocess_frame(self, frame):
    """Convert a dialog frame into a textual representation."""
    frame_str_parts = []
    service = frame["service"]
    if self.add_slot_desc:
      frame_str_parts.append(self.service_str_repr[service])
    else:
      frame_str_parts.append(f"service_name={service}")
    for action in frame["actions"]:
      action_str = self.preprocess_action(action, service)
      frame_str_parts.append(action_str)
    return SEPARATOR.join(frame_str_parts)

  def preprocess_action(self, action, service):
    """Convert a dialog action into a textual representation."""
    act = action["act"].lower()
    slot_name = action["slot"]
    values = ",".join(action["values"])
    action_str = act
    action_str_parts = [act]
    if slot_name:
      if self.add_slot_desc and (service, slot_name) in self.slots_str_repr:
        action_str_parts.append(self.slots_str_repr[(service, slot_name)])
      else:
        action_str_parts.append(slot_name)
    action_str = ",".join(action_str_parts)
    if values:
      action_str = f"{action_str}, values = {values}"
    return action_str

  def create_tsv_data(self, dialogs, output_tsv_path, query_dialog_ids=None):
    """Convert a dialog json file into a tsv for seq2seq models."""
    count = 0
    random.shuffle(dialogs)
    with tf.io.gfile.GFile(output_tsv_path, "w") as tsvfile:
      writer = csv.writer(tsvfile, delimiter="\t")
      for dialog in dialogs:
        dialog_id = dialog["dialogue_id"]
        if query_dialog_ids and dialog_id not in query_dialog_ids:
          continue
        services = dialog["services"]
        turns = dialog["turns"]
        # add turn number
        turn_id = 0
        for turn in turns:
          if turn["speaker"] == "USER":
            continue
          services = list(set([frame["service"] for frame in turn["frames"]]))
          if len(services) > 1 and self.input_representation_type != "t2g2":
            raise ValueError("found turn with multiple services. exiting.")
          service = services[0]
          schema = self.load_schema(self.schemas_path,
                                    service) if FLAGS.delexicalize else None
          text = _remove_newline_and_tabs(
              self.preprocess_target_utterance(turn, schema))
          structured_data = self.preprocess_turn(turn, schema)
          structured_data = _remove_newline_and_tabs(structured_data)
          metadata = _remove_newline_and_tabs(json.dumps(turn))
          writer.writerow([structured_data, text, metadata, dialog_id, turn_id])
          turn_id += 1


def create_fewshot_splits(dialogs, data_processor, output_dir, encoding_scheme):
  """Create fewshot splits for the training set."""
  for filename in os.listdir(FEWSHOT_IDS_DIR):
    dialog_ids = set()
    with tf.io.gfile.GFile(os.path.join(FEWSHOT_IDS_DIR, filename)) as f:
      for line in f:
        dialog_ids.add(line.strip())
    data_size = filename[:-4]  # Remove the .txt extension.
    tsv_path = os.path.join(output_dir, f"{encoding_scheme}_{data_size}.tsv")
    data_processor.create_tsv_data(dialogs, tsv_path, dialog_ids)


def main(_):
  # Process the train, dev and test splits.
  for split in ["train", "dev", "test"]:
    dialogs = []
    dataset_dir = os.path.join(FLAGS.sgd_dir, split)
    for filename in os.listdir(dataset_dir):
      if filename.startswith("schema"):
        continue
      with tf.io.gfile.GFile(os.path.join(dataset_dir, filename)) as f:
        dialogs.extend(json.load(f))

    for encoding_scheme in ["naive", "schema_guided", "t2g2"]:
      # Create data processor.
      schema_path = os.path.join(dataset_dir, "schema.json")
      data_processor = Preprocessor(schema_path, encoding_scheme)

      # Create the tsv file containing data from all dialogues in the split.
      output_dir = os.path.join(FLAGS.output_dir, split)
      if not tf.io.gfile.isdir(output_dir):
        tf.io.gfile.makedirs(output_dir)
      output_tsv_path = os.path.join(output_dir, f"{encoding_scheme}_all.tsv")
      data_processor.create_tsv_data(dialogs, output_tsv_path)

      # Create fewshot splits for the training set.
      if split == "train":
        create_fewshot_splits(dialogs, data_processor, output_dir,
                              encoding_scheme)


if __name__ == "__main__":
  app.run(main)
