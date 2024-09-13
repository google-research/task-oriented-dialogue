# AnyTOD: A Programmable Task-Oriented Dialog System

This directory contains the source code for [AnyTOD: A Programmable Task-Oriented Dialog System](https://arxiv.org/abs/2212.09939).

To cite this paper:

```
@misc{zhao2023anytod,
      title={AnyTOD: A Programmable Task-Oriented Dialog System}, 
      author={Jeffrey Zhao and Yuan Cao and Raghav Gupta and Harrison Lee and Abhinav Rastogi and Mingqiu Wang and Hagen Soltau and Izhak Shafran and Yonghui Wu},
      year={2023},
      eprint={2212.09939},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Generating Data

### SGD

```
python -m zero_shot_task_oriented_dialog.anytod.create_sgd_anytod_data \
  --input_dir="$SGD_INPUT_DIR" \
  --output_dir="$OUTPUT_DIR" \
  --mode=zombie_history_2pass \
  --cat_slots \
  --shuffle \
  --fix_tags
```


# Notes

 - "Zombie" was an internal name for AnyTOD, and we use this name throughout
   the source code.
