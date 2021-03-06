import utils.utils as utils

import numpy as np
import theano
import theano.tensor as T

class QFunction():
    """This is a class of modeling Q-value
    
    This class contain the weights and computation graph for Q-value.
    
    Attributes: 
        input_width(int): The #width of input.
        input_height(int): The #height of input.
        input_channel(int): The #channel of input.
        output_dim(int): The # of output dimension.
        
        conv1_window_size(int): The # of window size for the 1st convolution layer.
        conv1_stride(int): The # of stride for the 1st convolution layer.
        conv1_filter(int): The # of filter for the 1st convolution layer.
        conv2_window_size(int): The # of window size for the 2nd convolution layer.
        conv2_stride(int): The # of stride for the 2nd convolution layer.
        conv2_filter(int): The # of filter for the 2nd convolution layer.
        conv3_window_size(int): The # of window size for the 3rd convolution layer.
        conv3_stride(int): The # of stride for the 3rd convolution layer.
        conv3_filter(int): The # of filter for the 3rd convolution layer.
        full_in_dim(int): The # of dimension for the input of full connected layer(output of the 3rd convolution layer).
        full_out_dim(int): The # of dimension for the output of full connected layer.
        
        weights(dict): The dictionary record the theano shared variables of the network's weights.
        
        input: The theano variable of the input with shape (#batch, input_channel, input_width, input_height).
        action: The theano variable of the action taken with shape (#batch).
        output: The theano variable of the Q-value for each action with shape (#batch, output_dim).
        output_a: The theano variable of the Q-value of the action with shape (#batch).
        output_max: The theano variable of the max Q-value with shape (#batch).
        a_max: The theano variable of the action of max Q-value with shape (#batch).
    
    Args:
        input_width(int): The #width of input.
        input_height(int): The #height of input.
        input_channel(int): The #channel of input.
        output_dim(int): The # of output dimension.
    
    """
    def __init__(self, input_width, input_height, input_channel, output_dim):
        # input & output dimension
        self.input_width = input_width
        self.input_height = input_height
        self.input_channel = input_channel
        self.output_dim = output_dim
        # model size dimension
        self.conv1_window_size = 8
        self.conv1_stride = 4
        self.conv1_filter = 32
        self.conv2_window_size = 4
        self.conv2_stride = 2
        self.conv2_filter = 64
        self.conv3_window_size = 3
        self.conv3_stride = 1
        self.conv3_filter = 64
        conv1_width = (self.input_width + self.conv1_window_size - 1) / self.conv1_stride + bool((self.input_width + self.conv1_window_size - 1) % self.conv1_stride)
        conv1_height = (self.input_height + self.conv1_window_size - 1) / self.conv1_stride + bool((self.input_height + self.conv1_window_size - 1) % self.conv1_stride)
        conv2_width = (conv1_width + self.conv2_window_size - 1) / self.conv2_stride + bool((conv1_width + self.conv2_window_size - 1) % self.conv2_stride)
        conv2_height = (conv1_height + self.conv2_window_size - 1) / self.conv2_stride + bool((conv1_height + self.conv2_window_size - 1) % self.conv2_stride)
        conv3_width = (conv2_width + self.conv3_window_size - 1) / self.conv3_stride + bool((conv2_width + self.conv3_window_size - 1) % self.conv3_stride)
        conv3_height = (conv2_height + self.conv3_window_size - 1) / self.conv3_stride + bool((conv2_height + self.conv3_window_size - 1) % self.conv3_stride)
        self.full_in_dim = conv3_width * conv3_height * self.conv3_filter
        self.full_out_dim = 512
        
        # initialize the weights
        self.weights = self._init_weights()
        
        # build the computation graph
        self.input, self.action, self.output, self.output_a, self.output_max, self.a_max = self._build_forward()
    
    def set_weights(self, weights):
        """
        This is a method to set the weights of QFunction.
        
        Args:
            weights(dict): The weights dictionary to set QFunction.
        """
        for key in self.weights:
            if key not in weights:
                raise Exception("The weights can't be loaded to the Location network because %s is not in weights" % key)
            elif weights[key].shape != self.weights[key].get_value().shape:
                raise Exception("The weights can't be loaded to the Location network because of the shape error.")
            self.weights[key].set_value(weights[key])
            
    def get_weights(self):
        """
        This is a method to get the weights of QFunction.
        
        Returns:
            weights(dict): The weights dictionary of QFunction.
        """
        weights = {}
        for key in self.weights:
            weights[key] = self.weights[key].get_value()
        return weights
        
    def get_graph(self):
        """This method return the computation graph of QFunction."""
        return self.input, self.action, self.output, self.output_a, self.output_max, self.a_max, self.weights
    
    def _init_weights(self):
        """This is a helper method for __init__.
        
        This method initialize the theano shared weights.
        
        Returns: 
            shared_weights(dict): dictionary of theano shared variables to record the weights.
        """
        
        weights = {}
        shared_weights = {}
        
        # create the numpy weights
        weights['conv1_W'] = utils.norm_weight_4d(self.conv1_filter, self.input_channel, self.conv1_window_size, self.conv1_window_size)
        weights['conv1_b'] = utils.zero_bias(self.conv1_filter)
        weights['conv2_W'] = utils.norm_weight_4d(self.conv2_filter, self.conv1_filter, self.conv2_window_size, self.conv2_window_size)
        weights['conv2_b'] = utils.zero_bias(self.conv2_filter)
        weights['conv3_W'] = utils.norm_weight_4d(self.conv3_filter, self.conv2_filter, self.conv3_window_size, self.conv3_window_size)
        weights['conv3_b'] = utils.zero_bias(self.conv3_filter)
        weights['full_W'] = utils.norm_weight(self.full_in_dim, self.full_out_dim)
        weights['full_b'] = utils.zero_bias(self.full_out_dim)
        weights['output_W'] = utils.norm_weight(self.full_out_dim, self.output_dim)
        weights['output_b'] = utils.zero_bias(self.output_dim)
        
        # convert the numpy weights to theano shared variable
        for key, value in weights.iteritems():
            shared_weights[key] = theano.shared(value, name=key)
        
        return shared_weights
    
    def _build_forward(self):
        """This is a helper method for __init__.
        
        This method build the theano computation graph of Q function from input to output.
        
        Returns:
            [input, action, output]: Theano variable.
                input: Represent the input with shape (#batch, input_channel, input_width, input_height).
                action: Represent the action taken with shape (#batch).
                output: Represent the Q-value for each action with shape (#batch, output_dim).
                output_a: Represent the Q-value of the action with shape (#batch).
                output_max: Represent the max Q-value with shape (#batch).
                a_max: Represent the action of max Q-value with shape (#batch).
        """
        input = T.tensor4('input')
        action = T.ivector('action')
    
        conv1 = T.nnet.conv2d(input, self.weights['conv1_W'], border_mode='full', subsample=(self.conv1_stride, self.conv1_stride)) + self.weights['conv1_b'][None, :, None, None]
        act1 = T.nnet.relu(conv1)
        conv2 = T.nnet.conv2d(act1, self.weights['conv2_W'], border_mode='full', subsample=(self.conv2_stride, self.conv2_stride)) + self.weights['conv2_b'][None, :, None, None]
        act2 = T.nnet.relu(conv2)
        conv3 = T.nnet.conv2d(act2, self.weights['conv3_W'], border_mode='full', subsample=(self.conv3_stride, self.conv3_stride)) + self.weights['conv3_b'][None, :, None, None]
        act3 = T.nnet.relu(conv3)
        flat = T.reshape(act3, (act3.shape[0], -1))
        full = T.dot(flat, self.weights['full_W']) + self.weights['full_b']
        act4 = T.nnet.relu(full)
        output = T.dot(act4, self.weights['output_W']) + self.weights['output_b']
        
        output_a = output[T.arange(output.shape[0]), action]
        output_max, a_max = T.max_and_argmax(output,axis=1)
        
        return input, action, output, output_a, output_max, a_max 