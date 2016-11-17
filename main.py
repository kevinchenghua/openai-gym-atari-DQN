import argparse
import gym

import agent

def parse_args(args):
    parser = argparse.ArgumentParser(description="Run DQN training or evaluate process.")
    parser.add_argument('--env', dest='env', default=None, 
                        help="Atari environment of openai gym to run. e.g. Assault-v0")
    parser.add_argument('-r', '--is_reload', dest='is_reload', type=bool, default=True
                        help="Whether to reload the DQN trained.")
    parser.add_argument('-t', '--is_train', dest='is_train', type=bool, default=True
                        help="Whether to train the DQN.")
    parameters = parser.parse_args(args)
    if parameters.env is None:
        raise Exception("No atari environment found. Please specify it with: main.py --env atari_to_play")
    
    env = gym.make(parameters.env)
    
    a = agent.Agent(env, parameters.env, parameters.is_train, parameters.is_reload)
    
if __name__ == '__main__':
    parse_args(sys.argv[1:])