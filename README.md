<h1 align="center">Genomic Embedding Benchmark</h1>

<p align="center">
    <a href="https://github.com/tattabio/geb/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/v/release/tattabio/geb.svg">
    </a>
    <a href="">
        <img alt="arXiv URL" src="">
    </a>
    <a href="https://github.com/tattabio/geb/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/tattabio/geb.svg">
    </a>
    <a href="https://pepy.tech/project/geb">
        <img alt="Downloads" src="https://static.pepy.tech/personalized-badge/geb?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="#installation">Installation</a> |
        <a href="#usage">Usage</a> |
        <a href="https://huggingface.co/spaces/GEB">Leaderboard</a> |
        <a href="#documentation">Documentation</a> |
        <a href="#citing">Citing</a>
    <p>
</h4>

<h3 align="center">
    <a href="https://huggingface.co/spaces/GEB"><img style="float: middle; padding: 10px 10px 10px 10px;" width="100" height="100" src="./docs/images/tatta_logo.png" /></a>
</h3>

## Installation

TODO(joshua):
```bash
pip install geb
```

## Usage

- Using the python script (see [run_geb.py](https://github.com/tattabio/geb/blob/main/run_geb.py)):

```bash
python run_geb.py --model facebook/esm2_t6_8M_UR50D
```


- Using the python API:

```py
import geb

model = geb.get_model("facebook/esm2_t6_8M_UR50D")
tasks = geb.get_tasks_by_modality(geb.Modality.PROTEIN)
evaluation = geb.GEB(tasks=tasks)
evaluation.run(model, output_folder="results")
```


### Using a custom model

Custom models should be wrapped with the `geb.models.BioSeqTransformer` abstract class, and specify the modality, number of layers, and embedding dimension. See see [models.py](https://github.com/tattabio/geb/blob/main/geb/models.py) for additional examples on custom model loading and inference.


```python
import geb
from geb.models import BioSeqTransformer
from geb.modality import Modality

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


model = MyModel()
tasks = geb.get_tasks_by_modality(model.modality)
evaluation = MTEB(tasks=tasks)
evaluation.run(model)
```

### Evaluating on a custom dataset

TODO(andre): Update this section 

To evaluate on a custom task, you can run the following code on your custom task.

```python
import geb
from geb.tasks import AbsTask

class MyCustomTask(AbsTask):
    def run(
        self, model: BioSeqTransformer, layers: Optional[List[int]] = None
    ) -> TaskResult:
        pass

model = geb.models.ESM("facebook/esm2_t6_8M_UR50D")
evaluation = geb.GEB(tasks=[MyCustomTask()])
evaluation.run(model)
```

</details>

## Citing

GEB was introduced in "[GEB: Genomic Embedding Benchmark]()", feel free to cite:

TODO(andre): bibtex

For works that have used GEB for benchmarking, you can find them on the [leaderboard](https://huggingface.co/spaces/tattabio/GEB/leaderboard).
