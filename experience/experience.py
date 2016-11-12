from collections import deque
from scipy.ndimage.interpolation import zoom
import numpy as np
import sys


class Experience():
    def __init__(self, env, memory_size=10000, history_length=4, state_width=84, state_height=84, discount=0.99):
        # environment of atari game
        self.env = env
        # parameters for experience
        self.memory_size = memory_size
        self.history_length = history_length
        self.state_width = state_width
        self.state_height = state_height
        self.discount = discount
        # initialize the memory
        self.memory = self._init_memory()
    
    def preprocess(self, obs):
        """This is a method for state preprocess.
        
        This method convert staked observations to state.
        
        Args:
            obs(deque): The staked observations inputted with length #history_length.
        
        Returns:
            state: The preprocessed state.
        """

        if obs[-1] is None:
            state = None
        else:
            gray = np.array(obs).mean(3)
            gray_width = float(gray.shape[1])
            gray_height = float(gray.shape[2])
            state = zoom(gray,[1, self.state_width/gray_width, self.state_height/gray_height])
        return state
        
    def sample(self, size):
        """This is a method for sampling memory data.
        
        This method sample #size memory data with replacement.
        
        Args:
            size(int): The number of data to be sampled.
        
        Returns:
            samples(list): The list contain #size of Transition data.
        """
        index = np.random.randint(self.memory_size, size=size)
        samples = []
        for i in index:
            samples.append(self.memory[i])
        return samples
        
    def update(self, new_transition):
        """This method update the memory with new transition."""
        self.memory.append(new_transition)
        
    def _init_memory(self):
        """This is a helper method for __init__.
        
        This method initialize the memory with size #memory_size.
        
        Returns:
            memory(deque): The replay memory for training with content Transition.
        """
        memory = deque(maxlen=self.memory_size)
        while len(memory) < self.memory_size:
            # initialize the game and state
            obs = deque(maxlen=self.history_length)
            obs.append(self.env.reset())
            for i in range(self.history_length):
                action = self.env.action_space.sample()
                ob, reward, done, info = self.env.step(action)
                if done:
                    break
                obs.append(ob)
            if done:
                continue
            # loop for recording transitions to the memory
            while not done and len(memory) < self.memory_size:
                state = self.preprocess(obs)
                action = self.env.action_space.sample()
                ob, reward, done, info = self.env.step(action)
                obs.append(ob)
                state_next = self.preprocess(obs)
                memory.append(Transition(state, reward, action, state_next, self.discount))
                sys.stdout.write("Initialize the memory: %d / %d \r" % (len(memory), self.memory_size))
                sys.stdout.flush()
        return memory
    

    
class Transition():
    def __init__(self, s, r, a, s_next, discount):
        self.state = s
        self.reward = r
        self.action = a
        self.state_next = s_next if s_next is not None else s
        self.discount = discount if s_next is not None else 0