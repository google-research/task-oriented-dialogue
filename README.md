# Schema-Guided Dialogue

Virtual assistants such as the Google Assistant, Alexa, Siri, Cortana etc. help
users accomplish tasks by providing a natural language interface to service
providers (backends/APIs). For the democratization of such assistants, it is
important to seamlessly support an ever-increasing number of services and APIs.
In order to do so, a virtual assistant needs to have the following qualities:

* **Scalable** - Easy to support new APIs without requirement of skilled
  developers. Massively multidomain.
* **Low Maintenance** - Seamless addition of new APIs without retraining. Robust
  to addition of new entities in the API or changes in API's interface.
* **Data Efficient** - Massive parameter sharing. Reduced requirement of complex
  human-generated annotations.

This repository is a collection of research projects aimed at building a
data-driven virtual assistant with the above mentioned qualities.

**This is not an officially supported Google product. All resources in this
repository are  provided "AS IS" without any warranty, express or implied.
Google disclaims all liability for any damages, direct or indirect, resulting
from their use.**

## Dataset

We created the [Schema-Guided Dialogue Dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
to benchmark the above qualities of a virtual assistant. The dataset includes
annotated multi-domain conversations involving 45 APIs spanning across 20
domains. The evaluation sets contain some domains and APIs which are not present
in the training set in order to test the zero-shot generalization capability of
the dialogue systems.


## Language Understanding and Dialogue State Tracking

The schema-guided LU and DST model uses natural langauge description of slots
and intents to obtain their embedded semantic representation. Then a single
unified model is employed to make predictions conditioned on these embeddings.
With the use of BERT, the model is capable of zero-shot generalization to APIs
which are not present in the training set.

### Baseline Model

The baseline model for LU and DST was released along with the Schema-Guided
Dialogue Dataset. The model is described in [our paper](https://arxiv.org/pdf/1909.05855.pdf)
and code is available [here](https://github.com/google-research/google-research/tree/master/schema_guided_dst).
The code will be moved to this repository soon.

### Dialogue System Technology Challenge (DSTC)

The Schema-Guided State Tracking track in the [8th DSTC](https://dstc8.dstc.community/)
focussed on improving the baseline model for LU and DST. The participants were
able to significantly improve the performance of the baseline model while
reducing the gap in performance between seen and unseen APIs. More details about
the competition and the submissions may be found in the
[overview paper](https://arxiv.org/pdf/2002.01359.pdf).


## Natural Language Generation

[Our paper](https://www.aclweb.org/anthology/2020.emnlp-main.527.pdf)
investigates three different apporaches to building a multi-domain NLG system
without any domain or API dependent parameters, by using a pre-trained
[T5](https://github.com/google-research/text-to-text-transfer-transformer)
model. The code may be found in the directory `generation/`.



## Citations

```
@inproceedings{rastogi2019scalable,
  title={{Towards Scalable Multi-domain Conversational Agents: The Schema-Guided Dialogue Dataset}},
  author={Abhinav Rastogi and Xiaoxue Zang and Srinivas Sunkara and Raghav Gupta and Pranav Khaitan},
  year={2019},
  booktitle={{Proceedings of the AAAI Conference on Artificial Intelligence}},
  url="https://doi.org/10.1609/aaai.v34i05.6394",
}

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
