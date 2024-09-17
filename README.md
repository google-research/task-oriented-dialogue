# Task Oriented Dialogue

Repository for all open-sourced task oriented dialogue (TOD) research at Google.

**This is not an officially supported Google product. All resources in this
repository are  provided "AS IS" without any warranty, express or implied.
Google disclaims all liability for any damages, direct or indirect, resulting
from their use.**

## Updates

**09/16/2024** - Open source the AnyTOD dataset. Sorry for the delay :)

**03/28/2023** - Open source the STARv2 dataset. Code for the AnyTOD paper to be open-sourced soon.

**04/12/2022** - Open source the D3ST and SDT papers.

**04/04/2022** - Generalize this repository for all task-oriented dialogue
research being purused at Google. Originally, this repository held projects
related to the SGD dataset.

## Papers in this Repository

 - `end2end`:
    - `anytod`: AnyTOD: A Programmable Task-Oriented Dialog System ([Zhao et. al, 2022](https://arxiv.org/abs/2212.09939))
 - `generation`: Template Guided Text Generation ([Kale et. al, 2020](https://arxiv.org/abs/2004.15006))
 - `state_tracking`:
    - `d3st`: Description-Driven Dialogue State Tracking ([Zhao et. al, 2022](https://arxiv.org/abs/2201.08904))
    - `sdt`: Show Don't Tell ([Gupta et. al, 2022](https://arxiv.org/abs/2204.04327))
 - `starv2`: STARv2 datset ([Zhao et. al, 2022](https://arxiv.org/abs/2212.09939))

## Other Work

 - [Schema-Guided Dialogue Dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) ([Rastogi et. al, 2020](https://arxiv.org/pdf/1909.05855.pdf))
    - The source code for the baseline model released in SGD paper can be found
      [here](https://github.com/google-research/google-research/tree/master/schema_guided_dst).
    - The Schema-Guided State Tracking track in the [8th DSTC](https://dstc8.dstc.community/)
focussed on improving the baseline model for LU and DST. The participants were
able to significantly improve the performance of the baseline model while
reducing the gap in performance between seen and unseen APIs. More details about
the competition and the submissions may be found in the
[overview paper](https://arxiv.org/pdf/2002.01359.pdf).
