# add introduction

## Setting up environment

Run the following to create your ` conda environment

`conda create -n c4 python=3.10`

`pip install -r requirements.txt`


## Experimentation

Train:

```bash
python -m src.train
```

Evaluation and watching an trained agent play against itself:

```bash
python -m src.train \
  --watch \
  --resume-path log/connect4/dqn/policy.pth \
  --opponent-path log/connect4/dqn/policy.pth
```

Serving Tensorboard logs:

`tensorboard --logdir=log/connect4`