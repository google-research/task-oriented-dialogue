# "Show, Don't Tell" Modeling

This directory contains the source code for the paper "Show, Don't Tell: Demonstrations Outperform Descriptions for Schema-Guided Task-Oriented Dialogue" ([NAACL-22](https://aclanthology.org/2022.naacl-main.336/), [arxiv](https://arxiv.org/abs/2204.04327)).

To cite this paper:

```
@inproceedings{gupta-etal-2022-show,
    title = "Show, Don{'}t Tell: Demonstrations Outperform Descriptions for Schema-Guided Task-Oriented Dialogue",
    author = "Gupta, Raghav and Lee, Harrison and Zhao, Jeffrey and Cao, Yuan and Rastogi, Abhinav and Wu, Yonghui",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.336",
    doi = "10.18653/v1/2022.naacl-main.336",
    pages = "4541--4549",
}
```

## Generating Data

### Environment Setup

Create an virtual environment and install dependencies.

```
python3 -m venv default_env
source default_env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r task_oriented_dialogue/state_tracking/show_dont_tell/requirements.txt
```

### SGD

Example command:

```
python3 -m task_oriented_dialogue.state_tracking.show_dont_tell.create_sgd_sdt_data \
  --input_dir=dstc8-schema-guided-dialogue \
  --output_path=sgd_sdt_v0_train.tsv \
  --prompt_indices=0 \
  --mcq_cat_vals \
  --subdirs=train
```

This command produces a TSV file for text-to-text model training. The columns are as follows: source, target, dialogue_id, turn_id, and frame_id.

This command should be run once for each subdir - train, dev, and test. `prompt_indices` picks which demonstration prompt(s) from the `sdt_prompts.py` file to use. In our paper, we run 5 trials for indices 0-4 and report the mean.

`convert_sgd_t5x_sdt_preds_to_dstc8.py` is useful if you use [T5X](https://github.com/google-research/t5x). This script converts the T5X predictions to the official DSTC8 competition format, which can then be evaluated with scripts in this github repo: https://github.com/google-research/google-research/tree/master/schema_guided_dst

### MultiWOZ

Example command:

```
python3 -m task_oriented_dialogue.state_tracking.show_dont_tell.create_multiwoz_sdt_data \
  --input_dir=mw21_trade \
  --output_dir=mw21_trade_sdt/no_restaurant_v0 \
  --schema_file=task_oriented_dialogue/state_tracking/mw_schema.json \
  --multiwoz_version=2.1 \
  --is_trade \
  --use_active_domains_only \
  --blocked_domains=restaurant \
  --mcq_cat_vals \
  --prompt_indices=0
```

This command creates train, dev, and test TFRecord files, with these fields: input (source), value (target), dialog_id, and turn. In the command above, examples are created from all domains except for "restaurant". This is used for training and evaluating the MultiWOZ zero-shot/leave-one-domain-out setup. To generate the corresponding evaluation dataset, replace `--blocked_domains=restaurant` with `--blocked_domains=hotel,taxi,train,attraction,hospital,bus`.

Note: For MultiWOZ 2.1, we use the preprocessing from the
[TRADE paper](https://github.com/jasonwu0731/trade-dst), as per the
[official leaderboard](https://github.com/budzianowski/multiwoz#dialog-state-tracking).
