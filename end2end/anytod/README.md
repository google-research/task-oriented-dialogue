# AnyTOD: A Programmable Task-Oriented Dialog System

This directory contains the source code for [AnyTOD: A Programmable Task-Oriented Dialog System](https://arxiv.org/abs/2212.09939).

To cite this paper:

```
@misc{zhao2023anytod,
      title={AnyTOD: A Programmable Task-Oriented Dialog System}, 
      author={Jeffrey Zhao and Yuan Cao and Raghav Gupta and Harrison Lee and Abhinav Rastogi and Mingqiu Wang and Hagen Soltau and Izhak Shafran and Yonghui Wu},
      year={2022},
      eprint={2212.09939},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Generating Data

### STARv2

```
python -m zero_shot_task_oriented_dialog.anytod.create_starv2_anytod_data \
  --input_dir="$STARV2_DATA_DIR" \
  --output_dir="$STARV2_OUTPUT_DIR" \
  --mode=zombie_history_2pass \
  --use_cat_slots \
  --fix_tags
```

### SGD

We modified the SGD schemas to include an additional field `offered_slots` that
provided the slots possibly offered by the user during an intent. This
information was not saved in the original schemas provided in the dataset
release but did exist when we created the dataset.

To use these, copy the schema files in `sgd/` to overwrite the default schemas
in the SGD dataset.

```
cp sgd_schemas/train/schema.json $SGD_DATA_DIR/train/schema.json
cp sgd_schemas/dev/schema.json $SGD_DATA_DIR/dev/schema.json
cp sgd_schemas/test/schema.json $SGD_DATA_DIR/test/schema.json
```

Then, run the following command:

```
python -m zero_shot_task_oriented_dialog.anytod.create_sgd_anytod_data \
  --input_dir="$SGD_DATA_DIR" \
  --output_dir="$SGD_OUTPUT_DIR" \
  --mode=zombie_history_2pass \
  --cat_slots \
  --shuffle \
  --fix_tags
```

## Generating data to be labeled by D3ST

To create the STARv2 dataset with additional annotations, we:

1) Generate some examples from STAR for hand-labeling:

```
python -m zero_shot_task_oriented_dialog.create_star_data_for_d3st_labeling \
  --num_exs_per_task=5 --use_cat_slots --include_target_string=true
```

2) Hand-label the examples from above, and train a D3ST model from [Description-Driven Task Oriented Dialog Modeling](https://arxiv.org/abs/2201.08904) using SGD and these hand-labeled examples.

3) From this D3ST model, run predict over all STAR examples.

```
python -m zero_shot_task_oriented_dialog.create_star_data_for_d3st_labeling \
  --num_exs_per_task=1000000 --use_cat_slots --include_target_string=false
```

# Notes

 - "Zombie" was an internal name for AnyTOD, and we use this name throughout
   the source code.
