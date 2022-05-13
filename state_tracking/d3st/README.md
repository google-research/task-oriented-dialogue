# Description-Driven Task Oriented Dialog Modeling

This directory contains the source code for
[Description-Driven Task Oriented Dialog Modeling](https://arxiv.org/abs/2201.08904).

To cite this paper:

```
@article{zhao2022description,
  title={Description-Driven Task-Oriented Dialog Modeling},
  author={Zhao, Jeffrey and Gupta, Raghav and Cao, Yuan and Yu, Dian and Wang,
    Mingqiu and Lee, Harrison and Rastogi, Abhinav and Shafran, Izhak and Wu,
    Yonghui},
  journal={arXiv preprint arXiv:2201.08904},
  year={2022}
}
```

## Generating Data

### SGD

```
python -m zero_shot_task_oriented_dialog.d3st.create_sgd_schemaless_data \
  --sgd_file="$SGD_FILE" \
  --schema_file="$SCHEMA_FILE" \
  --output_file="$OUTPUT_FILE" \
  --delimiter== \
  --level=dst_intent \
  --data_format=full_desc \
  --multiple_choice=1a
```

Note that this command is per-file --- the output then needs to be combined.

### MultiWOZ

For MultiWOZ 2.2, 2.3, or 2.4:

```
python -m zero_shot_task_oriented_dialog.d3st.create_multiwoz_schemaless_data \
  --multiwoz_dir="$MULTIWOZ_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --schema_file="$SCHEMA_FILE" \
  --description_type=full_desc_with_domain \
  --delimiter== \
  --multiple_choice=1a
```

For MultiWOZ 2.1, we use the preprocessing from the
[TRADE paper](https://github.com/jasonwu0731/trade-dst), as per the
[official leaderboard](https://github.com/budzianowski/multiwoz#dialog-state-tracking).
Use the `zero_shot_task_oriented_dialog.d3st.create_multiwoz_schemaless_data`
module with the same arguments, except pointing to a TRADE-preprocessed 2.1
dataset.
