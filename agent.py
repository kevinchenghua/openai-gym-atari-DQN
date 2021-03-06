from collections import deque
from approximation.q_function import QFunction
from utils.utils import itemlist
from optimizer.optimizers import rmsprop
from experience.experience import Experience, Transition

import os
import sys
import numpy as np
import theano
import theano.tensor as T

theano.config.floatX = 'float32'

class Agent():
    def __init__(self, env, name, is_train=True, is_reload=True):
        # environment
        self.name = name
        self.env = env
        self.is_train = is_train
        self.is_reload = is_reload
        # model parameters
        self.input_width = 84
        self.input_height = 84
        self.input_channel = 4
        self.output_dim = self.env.action_space.n
        # memory parameters
        self.memory_size = 10000
        self.evaluation_memory_size = 10000
        self.history_length = 4
        # training parameters
        self.episodes = 1000000
        self.discount = 0.99
        self.batch_size = 32
        self.evaluate_episodes = 10
        self.evaluate_batch_size = 500
        self.lr = 0.00025
        self.er_start = 1.              # starting exploration rate
        self.er_end = 0.1               # ending exploration rate
        self.er_frame = 100000          # frames of exploration from start to end
        self.target_update_frame = 1000 # frames between target updates
        self.checkpoint_frame = 5000
        
        # initialize the Q function, action choosing function and updating function
        self.Q, self.Q_target = self._init_Q(self.input_width, self.input_height, self.input_channel, self.output_dim)
        self.f_action = self._build_Q_action_choosing()
        self.f_grad_shared, self.f_update, self.f_Q_max = self._build_updation()
        
        # initialize the experience memory
        self.memory = Experience(env, self.memory_size, self.history_length, self.input_width, self.input_height, self.discount)
        self.evaluation_memory = Experience(env, self.evaluation_memory_size, self.history_length, self.input_width, self.input_height, self.discount)
        
        # reload weights
        if self.is_reload and os.path.isfile('result/%s/log.npz'%(self.name)):
            self.load_weights('result/%s/log.npz'%(self.name))
        
        # training the agent
        if self.is_train:
            self._train(self.history_length, self.episodes, self.batch_size, self.evaluate_batch_size, 
                        self.lr, self.er_start, self.er_end, self.er_frame, self.target_update_frame, self.checkpoint_frame)
    
    def _init_Q(self, input_width, input_height, input_channel, output_dim):
        """This is a helper method for __init__.
        
        This method initialize the of action-value function Q and target action-value function Q_target.
        
        Returns:
            Q(QFunction): Action-value function.
            Q_target(QFunction): Target action-value function.
        """
        
        Q = QFunction(input_width, input_height, input_channel, output_dim)
        Q_target = QFunction(input_width, input_height, input_channel, output_dim)
        Q_target.set_weights(Q.get_weights())
        
        return Q, Q_target
    
    def _build_Q_action_choosing(self):
        """This is a helper method for __init__.
        
        This method build functions to choose action according to the max Q value.
        
        Returns:
            f_acion(function): state -> action
                state(float32 numpy array): The state input with shape (#batch, input_channel, input_width, input_height).
        """
        input, _, _, _, _, a_max, _ = self.Q.get_graph()
        f_action = theano.function([input], a_max)
        return f_action
    
    def _build_updation(self):
        """This is a helper method for __init__.
        
        This method build functions to compute gradient and update weights.
        
        Returns:
            f_grad_shared(function):[state, action, reward, state_next, discount] -> loss (with side-effect of updating shared gradient)
                state(float32 numpy array): The state input with shape (#batch, input_channel, input_width, input_height).
                action(int32 numpy array): The action input with shape (#batch).
                reward(float32 numpy array): The reward input with shape (#batch).
                state_next(float32 numpy array): The next state input with shape (#batch, input_channel, input_width, input_height).
                discount(float32 numpy array): The discount input with shape (#batch).
                loss(float): The loss output.
            f_update(function): lr -> (with side-effect of updating Q_weights)
                lr(float): The learning rate to update Q_weights.
            f_Q_max(function): [state] -> Q_max
                state(float32 numpy array): The state input with shape (#batch, input_channel, input_width, input_height).
                Q_max(float32 numpy array): The max Q value according to the state with shape (#batch)
        """
        discount = T.vector()
        reward = T.vector()
        lr = T.scalar()
        state, action, _, Q_a, Q_max, _, Q_weights = self.Q.get_graph()
        state_next, _, _, _, Q_target_next_max, _, _ = self.Q_target.get_graph()
        inputs = [state, action, reward, state_next, discount]
        
        target = reward + discount * Q_target_next_max
        loss = T.sqr(target - Q_a).mean()
        grads = T.grad(cost=None, wrt=itemlist(Q_weights), disconnected_inputs='raise', known_grads={Q_a: Q_a-target})
        
        f_grad_shared, f_update = rmsprop(lr, Q_weights, grads, inputs, loss)
        f_Q_max = theano.function([state], Q_max)
        
        return f_grad_shared, f_update, f_Q_max
        
    def _train(self, history_length, episodes, batch_size, evaluate_batch_size, lr, er_start, er_end, er_frame, target_update_frame, checkpoint_frame):
        if self.is_reload and os.path.isfile('result/%s/log.npz'%(self.name)):
            with np.load('result/%s/log.npz'%(self.name)) as log:
                self.env.monitor.start('result/%s'%(self.name), resume=True)
                frame = log['frame']
                episode = log['episode']
                history_Q_ave = log['history_Q_ave'].tolist()
                history_loss = log['history_loss'].tolist()
                history_score = log['history_score'].tolist()
                episode_record = log['episode_record'].tolist()
        else:
            self.env.monitor.start('result/%s'%(self.name), force=True)
            frame = 0
            episode = 0
            history_Q_ave = []
            history_loss = []
            history_score = []
            episode_record = []
        for i in range(episode, episodes):
            # initialize the episode
            print "%d episode start." % (i)
            obs = deque(maxlen=history_length)
            ob = self.env.reset()
            for j in range(history_length):
                obs.append(ob)
            done = False
            # start episode playing and training
            score = 0
            loss = np.array([])
            while not done:
                # play with one step
                #self.env.render()
                state = self.memory.preprocess(obs)
                if frame < er_end:
                    er_rate = er_start*(er_frame - frame)/er_frame + er_end*frame/er_frame
                else:
                    er_rate = er_end
                action = self.choose_action_e_greedy(er_rate, state)
                ob, reward, done, info = self.env.step(action)
                obs.append(ob)
                state_next = self.memory.preprocess(obs)
                self.memory.update(Transition(state, action, reward, state_next, self.discount))
                score += reward
                # train Q with a batch Transition
                states, actions, rewards, states_next, discounts = self.memory.sample(batch_size)
                loss = np.append(loss, self.f_grad_shared(states, actions, rewards, states_next, discounts))
                self.f_update(lr)
                frame += 1
                sys.stdout.write("Frame: %d \r" % (frame))
                sys.stdout.flush()
                # target Q update
                if frame % target_update_frame == 0:
                    self.Q_target.set_weights(self.Q.get_weights())
                # checkpoint
                if frame % checkpoint_frame == 0:
                    print "Saving weights...",
                    Q_weights = self.Q.get_weights()
                    np.savez('result/%s/log.npz'%(self.name), frame=frame, episode=i,
                                                              history_Q_ave=history_Q_ave, history_loss=history_loss, 
                                                              history_score=history_score, episode_record=episode_record, **Q_weights)
                    print 'done'
                    
            # display the result of this episodes
            print "Training episodes %d get score: %f" % (i, score)
            print "Average loss: %f" % (loss.mean())
            if i % self.evaluate_episodes == 0:
                states,_,_,_,_ = self.evaluation_memory.sample(evaluate_batch_size)
                Q_ave = self.f_Q_max(states).mean()
                history_Q_ave.append(Q_ave)
                history_loss.append(loss.mean())
                history_score.append(score)
                episode_record.append(i)
                print "Episode %d, evaluate average Q value: %f" % (i, Q_ave)
        self.env.monitor.close()
                
    def choose_action_e_greedy(self, er_rate, state):
        """This method return an action.
        
        This method choose an action according to e-greedy policy. The policy do exploration (same probability for each action) 
        with probability er_rate, and exploit according to Q value with probability (1-er_rate).
        
        Returns:
            action: Represent the action to send to the environment.
        """
        if np.random.rand() < er_rate:
            action = self.env.action_space.sample()
        else:
            action = self.f_action(state[None, :])[0]
        return action
        
    def load_weights(self, path):
        with np.load(path) as log:
            tmp = self.Q.get_weights()
            weights = {}
            for key in tmp:
                if key not in log:
                    raise Exception("The weights can't be loaded %s is not in weights" % key)
                elif log[key].shape != tmp[key].shape:
                    raise Exception("The weights can't be loaded because of the shape error of %s." % key)
                weights[key] = log[key]
            self.Q.set_weights(weights)
            self.Q_target.set_weights(weights)
            
    def evaluate(self):
        obs = deque(maxlen=self.history_length)
        ob = self.env.reset()
        for i in range(self.history_length):
            obs.append(ob)
        done = False
        score = 0
        while not done:
            self.env.render()
            state = self.memory.preprocess(obs)
            action = self.f_action(state[None, :])[0]
            ob, reward, done, info = self.env.step(action)
            obs.append(ob)
            score += reward