from collections import deque
import numpy as np

class Experience():
    def __init__(self, env, memory_size=100000, history_length=4, discount=0.99):
        # environment of atari game
        self.env = env
        # parameters for experience
        self.memory_size = memory_size
        self.history_length = history_length
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
            state = np.array(obs)
        return state
        
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
                action = env.action_space.sample()
                ob, reward, done, info = env.step(action)
                if done:
                    break
                obs.append(ob)
            if done:
                continue
            # loop for recording transitions to the memory
            while not done and len(memory) < self.memory_size:
                state = self.preprocess(obs)
                action = env.action_space.sample()
                ob, reward, done, info = env.step(action)
                obs.append(ob)
                state_next = self.preprocess(obs)
                memory.append(Transition(state, reward, action, state_next, self.discount))
        return memory
    

    
class Transition():
    def __init__(self, s, r, a, s_next, discount):
        self.state = s
        self.reward = r
        self.action = a
        self.state_next = s_next if s_next !=None else s
        self.discount = discount if s_next !=None else 0