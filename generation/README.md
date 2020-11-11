# Template Guided Text Generation

## Preparing tsv files

The following command generates tsv files containing complete train/dev/test
datasets and few shot splits for the training set for all the three encoding
schemes - *Naive*, *Schema Guided* and *T2G2*. The SGD dataset directory can be
downloaded from
[github](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue).

```
# Run this command with schema_guided_dialogue as current directory.

python -m generation.prepare_dataset --sgd_dir <sgd_dataset_directory> \
--output_dir <output_directory>
```
