# TensorFlow minGPT

It's pretty much the same as [Andrej Karpathy's GPT Tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY) but using TensorFlow instead of PyTorch. Setup and training:

```bash
conda env create -f conda.yml

conda activate tf-mingpt

python main.py --help
```

There are two commands:

1. `python main.py train --config-file shakespeare.yml` - Train a model
2. `python main.py generate --config-file shakespeare.yml` - Generate sequences from the trained model

Filepath to config files will need to be explicityly provided as the default is `superhero.yml` which creates a really small model. The `shakespeare.yml` config has the same parameter values as used in the tutorial above.

I'd suggest training with `superhero.yml` config if you have only a CPU.

If you want to train with the `shakespeare.yml` config, and don't have a GPU locally, I'd suggest setting up [SkyPilot](https://skypilot.readthedocs.io/en/latest/) and launching the provided SkyPilot task. E.g. `sky launch mingpt.yml` for training on shakespeare dataset and `sky launch mingpt.yml --env MODEL_PREFIX=superhero` for the smaller superhero dataset.

## Next

- Update model to achieve a loss of ~1.4 on validation (currently ~1.7)
- What is a test anyway? :/
