# Solving NP-Hard Problems on Graphs by Reinforcement Learning without Domain Knowledge
Pytorch implementation of the above paper

## Getting Started
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Directory Structure
- environ: graph environment (MDP)
- gcn: graph convolutional netoworks model
- gin: graph isomophism networks model
- policy: policy gradient method (REINFORCE)
- trainer: train with REINFORCE
- mcts: Monte Carlo Tree Search
- utils: utils

## Reference on Implementation
- https://github.com/tkipf/pygcn
- https://github.com/weihua916/powerful-gnns
