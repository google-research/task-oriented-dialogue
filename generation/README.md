# Template Guided Text Generation


## Usage

To run this code, you need to install the packages in
[requirements.txt](requirements.txt) including the
[t5 library](https://pypi.org/project/t5/). General instructions for training,
fine-tuning, evaluation, and exporting models for inference can be found in the
[t5 repo](https://github.com/google-research/text-to-text-transfer-transformer).


## Preparing TSV Files

The following command generates tsv files containing complete train/dev/test
datasets and few shot splits for the training set for all the three encoding
schemes - *Naive*, *Schema Guided* and *T2G2*. The SGD dataset directory can be
downloaded from
[github](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue).

```
# Run this command from the repository home directory.

python -m generation.prepare_dataset --sgd_dir <sgd_dataset_directory> \
--output_dir <output_directory>
```

Please update `TSV_DATA_DIR` in [t5\_tasks.py](t5_tasks.py) with the output
directory used in the above command.

## Fine-Tuning

Each of the experiments in the [paper][paper] is defined as a task in
[t5\_tasks.py](t5_tasks.py). The task names are formatted as
`<encoding_scheme>_<split>` with `naive`, `schema_guided` and `t2g2` as the
encoding schemes and `5_shot`, `10_shot`, `20_shot`, `40_shot`, `80_shot` and
`all` as the split.

As an example, to finetune the `T5-Small` model on the `t2g2_all` task:

```
export PROJECT=yourproject
export ZONE=yourzone
export BUCKET=yourbucket
export TPU=yourtpu

ctpu up --name=$TPU --project=$PROJECT --zone=$ZONE --tpu-size=v3-256 --tpu-only --noconf

TASK=t2g2_all
PRETRAINED_DIR=gs://t5-data/pretrained_models/small
PRETRAINED_STEPS=1000000
FINETUNE_STEPS=5000
MODEL_DIR="${BUCKET}${TASK}"

# Run fine-tuning
t5_mesh_transformer \
  --tpu="${TPU}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --model_dir="${MODEL_DIR}" \
  --gin_file="${PRETRAINED_DIR}/operative_config.gin" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = 'v3-256'" \
  --gin_param="MIXTURE_NAME = '${TASK}'" \
  --gin_param="utils.run.train_steps=$((PRETRAINED_STEPS+FINETUNE_STEPS))" \
  --gin_param="utils.run.init_checkpoint='${PRETRAINED_DIR}/model.ckpt-${PRETRAINED_STEPS}'" \
  --module_import="generation.t5_tasks" \
```


## How to Cite

If you extend or use this work, please cite the following [paper][paper]:

```
@inproceedings{kale-rastogi-2020-template,
    title = "Template Guided Text Generation for Task Oriented Dialogue",
    author = "Kale, Mihir  and Rastogi, Abhinav",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.527",
    pages = "6505--6520",
}
```
[paper]: https://www.aclweb.org/anthology/2020.emnlp-main.527.pdf
