# Hypertuning
### data
Pick an image dataset.
You could pick the flowers, or the ants/bees from notebook [03_minimal_transfer_learning.ipynb](https://github.com/raoulg/MADS-MachineLearning-course/blob/master/notebooks/4_tuning_networks/03_minimal_transfer_learning.ipynb), or even another set you like from [torchvision](https://pytorch.org/vision/0.8/datasets.html) or [torchtext](https://pytorch.org/text/stable/datasets.html#imdb).

Keep in mind that your hardware will limit what you can do with the images; so you will either need to settle for smaller images, or preprocess them into a smaller size, or use google colab if the VM is too slow or limited in memory.

## Create a model
We have worked with a few types of layers:
- linear
- conv2d
- dropout
- batchnorm

and we have also seen architectures build with these layers like resnet (skip-layers), squeeze-excite and googlenet (inception). If you dont know how to implement these, ask claude.ai or gemini for help, they do an excellent job at explaining these architectures.
It's up to you to create a configurable model now that can be hypertuned.

## Goal

Show you can
1. Make a hypothesis based on the theory (use the book)
1. Design experiments to test your hypothesis
1. Work iterative: eg start with a small experiment to get a direction, then reduce the search space and run a more extensive experiment

Some hyperparameters to consider, ranging from high to low importance:
- the architecture of the model is the most important. Number of layers, type of layers, skiplayers, etc.
- the number of hidden units directly impacts the number of learnable parameters. This determines the complexity the model can handle, and should relate to the complexity of the data.
- dropout, batchnorm, skiplayers all help to reduce overfitting, so it is more relevant for bigger models.
- learning rate, type of optimizer, type of activation function, etc.

!!!IMPORTANT!!!: Dont use hyperband when trying to create a heatmap! Because this will cause you to put models that have run just a few epochs together with models that have run many epochs. This will NOT give you a clear overview of the interaction between the variables.

You CAN use hyperband in the process (eg if you want to speed up scans of big hyperparameter spaces), but always keep in mind what you are doing when creating visualizations.

- Visualise your finding
- reflect on what you see, using the theory. Please dont use chatGPT to reflect: you will get text that makes you look like an idiot. Instead, try to make mistakes, it wont be subtracted from your grade; making mistakes gives me an oportunity to correct you, which might actually help you during your exam.

### Scientific method
The science part of data science is about setting up experiments. You follow this cycle:
- make a hypothesis
- design an experiment
- run the experiment
- analyze the results and draw conclusions
- repeat

To keep track of this process, it is useful to keep a journal. While you could use anything to do so, a nice command line tool is [jrnl](https://jrnl.sh/en/stable/). This gives you the advantage of staying in the terminal, just type down your ideas during the process, and you can always look back at what you have done.
First formulate a hypothesis, and then design an experiment to test it. You are encouraged to use both your book and use an LLM as a sparring partner (eg use gemini or claude. I advise against chatgpt) to help formulate your hypothesis, or to finetune it.

Important: the report you write is NOT the same as your journal! The journal will help you to keep track of your process, and later write down a reflection on what you have done where you draw conclusion, reflecting back on the theory.


## Implementing
In hypertune.py, I have set up an example for hypertuning.
Implement your own hypertuner for another model / dataset from scratch.

### environment setup
- dont blindly reuse the environment from the lessons; this means, DO NOT simply copy pyproject.toml but create a minimal environment by following [the codestyle](https://github.com/raoulg/codestyle/blob/main/docs/make_a_module.md)
- Check the [dependencies for mltrainer](https://github.com/raoulg/mltrainer/blob/main/pyproject.toml) to see whats already installed if you use mltrainer
- have a look at the pyproject.toml file from the course;
    - `configspace` is a dependency for `hpbandster`.
    - `bayesian-optimization` is for, well, bayesian optimization
    - `hyperopt` is another hyperoptimalisation library which we have used in `2_convolutions/03_mlflow.ipynb`. It doesnt has as much algorithms as ray, which is why we use ray. Just know that `hyperopt` can be an alternative to `ray.tune` that's slightly easier to setup but less powerful.

### split code and configuration
- make a *configurable model* where you can change the different options from the config. You can decide for yourself if you want to make it configurable by simply passing a dictionary, or by creating a @dataclass as a Settings object, or by using a `.toml` file with `tomllib` (since python 3.11 part of python) and the `TOMLserializer`.

## Report
Write a report of **maximum** 3 a4 pages; but 2 a4 should be enough. The master shows in knowing what to remove...
This means you will have to be very clear with your report. Remove all clutter. Use clear visualisations. Make sure it's clear in a few seconds what the results are of every experiment.

You will get a:
- 0, which means: you will have to improve things, otherwise it will cost you points at your exam
- 1, meaning: your work is fine. There are some things to improve, but you are on the right track for the final report.
- 2, meaning: your work is excellent. You have done everything right, even exceeded expectations,and you have shown that you understand the material. If you do the same on the exam you will be good, and even though i give improvements, this should be interpreted as a direction for future growth and not as a critique on your current level.

# Common gotchas
- you can't blindly copy paths. Always use `Path` and check if locations exist with `.exists()`
- If you want to add your own `src` folder for import, the best way is to properly create your environment with `uv`. Another option is to figure out if your `src` folder is located at the ".." or "../.." location, relative to your notebook, and use a `sys.path.insert(0, "..")` command (that is, if you need to explicitly add it because the notebook cant find your src folder)
- same goes for datalocations. `"../../data/raw"` might have changed into `"../data/raw"` depending in your setup. `mads_datasets` uses `Path.home() / ".cache/mads_datasets"` as a default location, so you can use that from every folder on your computer.
- PRACTICE linting and formating (using a Makefile makes this easier). ruff format and ruff check should be run, it helps you to improve your code. (ruff has a --fix argument to autofix issues), mypy often takes a bit more effort (because you need to typehint) but you will become a better programmer if you learn to think about the types of your input and output, and LLMs can help you with this. Additionally mypy will catch possible errors that dont show up during a first run (but might show up later, with different input)
# Hypertuning assignment — Ray Tune

This folder contains an example hyperparameter tuning setup using Ray Tune. Below are the assignment description, the implementation notes and exact commands to reproduce experiments.

## Assignment objective
- Design and run a hyperparameter search for an RNN model on the gestures dataset.
- Use Ray Tune to explore a configurable search space, document your experimental protocol, and produce a short English report with results and conclusions.

## Files in this folder
- `hypertune.py`: the implemented Ray Tune experiment (example trainer that uses `mltrainer` and `mads_datasets`).
- `instructions.md`: this file (assignment + run instructions).
- `summary.md`: short English report with hypothesis, setup and reproduction steps.

If you want to adapt the experiment, edit `hypertune.py` (search space and training details are defined there).

## Key design choices (implementation summary)
- Dataset: `mads_datasets` gestures dataset (see `hypertune.py` — `DatasetType.GESTURES`).
- Model family: GRU-based RNNs via `mltrainer.rnn_models` and `rnn_models.GRUmodel`.
- Search space (example in `hypertune.py`):
    - `hidden_size`: integer sampled from [16, 128]
    - `dropout`: uniform in [0.0, 0.3]
    - `num_layers`: integer sampled from [2, 4]
- Tuner: `HyperOptSearch` + `AsyncHyperBandScheduler` (the example uses `NUM_SAMPLES=50` and `MAX_EPOCHS=10`).

## Environment & dependencies
Use the project's virtual environment or create a new one. Minimal packages required (examples):

```bash
# create + activate venv (if not already active)
python -m venv .venv
source .venv/bin/activate

# install dependencies (adjust if you use poetry/uv)
pip install -U pip
pip install ray[default] torch filelock loguru mltrainer mads_datasets
```

Note: prefer installing from the project's `pyproject.toml` when available; the command above lists the main runtime requirements for `hypertune.py`.

## Running the experiment
From the project root, run:

```bash
python 4-hypertuning-ray/hypertune.py
```

This script calls `ray.init()` and `tune.run(...)`. Outputs (Ray storage) will be written under `logs/ray` by default (see `hypertune.py`). Model-level reports use `ReportTypes.RAY` so Ray receives metrics.

### Quick local test (single-sample)
To test quickly without running the full search, edit `hypertune.py` and set `NUM_SAMPLES = 1` and `MAX_EPOCHS = 2`, then run the same command. This will verify environment and dataloaders.

## Reproducing and analysing results
- After a full run the `tune.run` returns an `analysis` object.
- To retrieve the best config after a run, open a Python REPL in the project root and run:

```python
from ray import tune
from pathlib import Path
from ray.tune.analysis import Analysis

analysis = Analysis(str(Path('logs/ray')))
best_config = analysis.get_best_config(metric='test_loss', mode='min')
print(best_config)
```

- You can also load the results into a pandas DataFrame for plotting:

```python
df = analysis.dataframe(metric='test_loss')
```

## Notes & best practices
- Use a small pilot run to narrow the search space before committing to many samples.
- When creating heatmaps or interactive visualisations, avoid mixing results from early-stopped trials (HyperBand) with fully trained trials — filter by `training_iteration` or use experiments with the same `max_epochs`.
- Locking: `hypertune.py` already uses a `FileLock` to avoid simultaneous downloads of the gestures dataset.

## What to hand in (deliverables)
1. This folder contains your implementation (`hypertune.py`).
2. A short English report `summary.md` (placed in this folder) describing the hypothesis, experimental protocol, hyperparameter search space, key results and conclusions.
3. A reproducible command to rerun the best experiment and the list of package versions used.

If you want, I can run a quick pilot experiment on your machine and then update `summary.md` with the actual results.