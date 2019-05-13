# Training Graph Convolutional Networks by Reinforcement Learning for Solving NP-hard Problems
Pytorch implementation of the above paper

## Getting Started
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Directory Structure
- environ: graph environment (MDP)
- gcn: graph convolutional netoworks
- policy: predict one action from graph
- agent: get a solution (MIS) from graph based on policy
- trainer: train with REINFORCE

## Reference
- https://github.com/tkipf/pygcn
- "Semi-Supervised Classification with Graph Convolutional Networks"