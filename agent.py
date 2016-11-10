from approximation.q_function import QFunction
from utils.utils import itemlist
from optimizer.optimizers import rmsprop

import numpy
import theano
import theano.tensor as T

class Agent():
    def __init__(self, input_width, input_height, input_channel, output_dim):
        self.Q, self.Q_target = self._init_Q(input_width, input_height, input_channel, output_dim)
        self.f_grad_shared, self.f_update = self._build_updation()
    
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
            
    def _build_updation(self):
        """This is a helper method for __init__.
        
        This method this method build functions to compute gradient and update weights.
        
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
        """
        discount = T.vector()
        reward = T.vector()
        lr = T.scalar()
        state, action, _, Q_a, _, _, Q_weights = self.Q.get_graph()
        state_next, _, _, _, Q_target_next_max, _, _ = self.Q_target.get_graph()
        inputs = [state, action, reward, state_next, discount]
        
        target = reward + discount * Q_target_next_max
        loss = T.sqr(target - Q_a).mean()
        grads = T.grad(cost=None, wrt=itemlist(Q_weights), disconnected_inputs='raise', known_grads={Q_a: target-Q_a})
        
        f_grad_shared, f_update = rmsprop(lr, Q_weights, grads, inputs, loss)
        
        return f_grad_shared, f_update
        