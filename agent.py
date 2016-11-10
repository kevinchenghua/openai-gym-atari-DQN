from approximation.q_function import QFunction

class Agent():
    def __init__(self, input_width, input_height, input_channel, output_dim):
        self.Q, self.Q_target = self._init_Q(input_width, input_height, input_channel, output_dim)
    
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