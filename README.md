# openai-gym-atari-DQN
This is an implementation of deep Q-network (DQN) to play the atari games in [openai gym](https://gym.openai.com/envs#atari) (without RAM).

The algorithm is described in ["Human-level control through deep reinforcement learning"](http://home.uchicago.edu/~arij/journalclub/papers/2015_Mnih_et_al.pdf). This package is a Theano-based implementation of the algorithm.
## Requirements
* Python 2.7
* [Theano (bettet to use GPU)](http://deeplearning.net/software/theano/install.html#install)
* [gym[atari]](https://github.com/openai/gym)

## Usage
You can play all the atari game (without ram) list [here](https://gym.openai.com/envs#atari). Take `Assault-v0` as example, to train:
```
python main.py --env 'Assault-v0'
```
reload the agent and evaluate (play) without training:
```
python main.py --env 'Assault-v0' -t false
```

## Result
