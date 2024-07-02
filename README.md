<h1 align="center">Diverse Genomic Embedding Benchmark</h1>

<p align="center">
    <a href="https://github.com/tattabio/dgeb/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/v/release/tattabio/dgeb.svg">
    </a>
    <a href="">
        <img alt="arXiv URL" src="">
    </a>
    <a href="https://github.com/tattabio/dgeb/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/tattabio/dgeb.svg">
    </a>
    <a href="https://pepy.tech/project/dgeb">
        <img alt="Downloads" src="https://static.pepy.tech/personalized-badge/dgeb?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="#installation">Installation</a> |
        <a href="#usage">Usage</a> |
        <a href="https://huggingface.co/spaces/dgeb">Leaderboard</a> |
        <a href="#citing">Citing</a>
    <p>
</h4>

<h3 align="center">
    <a href="https://huggingface.co/spaces/dgeb"><img style="float: middle; padding: 10px 10px 10px 10px;" width="100" height="100" src="./docs/images/tatta_logo.png" /></a>
</h3>

## Installation

Currently, DGEB sits on the Test PyPI index. Here's the command to install:

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ dgeb
```

## Usage

- Launch evaluation using the python script (see [run_dgeb.py](https://github.com/tattabio/dgeb/blob/main/run_dgeb.py)):

```bash
python run_dgeb.py --model facebook/esm2_t6_8M_UR50D
```

- To see all supported models and tasks:

```bash
python run_dgeb.py --help
```


- Using the python API:

```py
import dgeb

model = dgeb.get_model("facebook/esm2_t6_8M_UR50D")
tasks = dgeb.get_tasks_by_modality(dgeb.Modality.PROTEIN)
evaluation = dgeb.DGEB(tasks=tasks)
evaluation.run(model, output_folder="results")
```

### Using a custom model

Custom models should be wrapped with the `dgeb.models.BioSeqTransformer` abstract class, and specify the modality, number of layers, and embedding dimension. See [models.py](https://github.com/tattabio/dgeb/blob/main/dgeb/models.py) for additional examples on custom model loading and inference.

```python
import dgeb
from dgeb.models import BioSeqTransformer
from dgeb.modality import Modality

class MyModel(BioSeqTransformer):

    @property
    def modality(self) -> Modality:
        return Modality.PROTEIN

    @property
    def num_layers(self) -> int:
        return self.config.num_hidden_layers

    @property
    def embed_dim(self) -> int:
        return self.config.hidden_size


model = MyModel(model_name='path_to/huggingface_model')
tasks = dgeb.get_tasks_by_modality(model.modality)
evaluation = MTEB(tasks=tasks)
evaluation.run(model)
```


### Evaluating on a custom dataset

**We strongly encourage users to contribute their custom datasets to DGEB. Please open a PR adding your dataset so that the community can benefit!**

To evaluate on a custom dataset, first upload your dataset to the [Huggingface Hub](https://huggingface.co/docs/hub/en/datasets-adding). Then define a `Task` subclass with `TaskMetadata` that points to your huggingface dataset. For example, a classification task on a custom dataset can be defined as follows:

```python
import dgeb
from dgeb.models import BioSeqTransformer
from dgeb.tasks import Dataset, Task, TaskMetadata, TaskResult
from dgeb.tasks.classification_tasks import run_classification_task

class MyCustomTask(Task):
    metadata = TaskMetadata(
        id="my_custom_classification",
        display_name="...",
        description="...",
        type="classification",
        modality=Modality.PROTEIN,
        datasets=[
            Dataset(
                path="path_to/huggingface_dataset",
                revision="...",
            )
        ],
        primary_metric_id="f1",
    )

    def run(self, model: BioSeqTransformer) -> TaskResult:
        return run_classification_task(model, self.metadata)

model = dgeb.get_model("facebook/esm2_t6_8M_UR50D")
evaluation = dgeb.DGEB(tasks=[MyCustomTask])
evaluation.run(model)
```


## Leaderboard

TODO(nishant): 
- Add link to leaderboard
- Instruction for uploading results

## Acknowledgements

DGEB follows the design of text embedding bechmark [MTEB](https://github.com/embeddings-benchmark/mteb) developed by Huggingface 🤗. The evaluation code is adapted from the MTEB codebase.

## Citing

DGEB was introduced in "[DGEB: Diverse Genomic Embedding Benchmark]()", feel free to cite:

TODO(andre): bibtex




