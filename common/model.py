from .misc_util import orthogonal_init
import torch
import numpy as np
import torch.nn as nn


def mlp_block(in_features, out_features, use_batchnorm):
    layers = []
    layer = orthogonal_init(nn.Linear(in_features, out_features),
                            gain = nn.init.calculate_gain('relu'))
    layers.append(layer)
    if use_batchnorm:
        layers.append(nn.BatchNorm1d(out_features))
    layers.append(nn.ReLU())
    
    return nn.Sequential(*layers)

class MlpModel(nn.Module):
    def __init__(self, 
                 input_dim   = 4, 
                 hidden_dims = [64, 64],
                 use_batchnorm = False,
                 **kwargs):
        """
        input_dim:     (int)  number of the input dimensions
        hidden_dims:   (list) list of the dimensions for the hidden layers  
        use_batchnorm: (bool) whether to use batchnorm
        """ 
        super(MlpModel, self).__init__()   
        
        # Hidden layers
        hidden_dims = [input_dim] + hidden_dims
        layers = []
        for i in range(len(hidden_dims)-1):
            in_features  = hidden_dims[i]
            out_features = hidden_dims[i+1]
            layers.append(mlp_block(in_features, out_features, use_batchnorm))
        self.layers = nn.Sequential(*layers)        
        self.output_dim = hidden_dims[-1]
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    
def conv_block(in_channels, out_channels, kernel_size, stride, use_batchnorm):
    layers = []
    layer = orthogonal_init(nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride = stride),
                            gain = nn.init.calculate_gain('relu'))
    layers.append(layer)
    if use_batchnorm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU())

    return nn.Sequential(*layers)
        
def conv2d_size_out(size, kernel_size, stride):
    return (size - (kernel_size - 1) - 1) // stride  + 1    

class ConvModel(nn.Module):
    def __init__(self, 
                 input_shape = (3, 84, 84), 
                 filters = [(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                 use_batchnorm = False,
                 **kwargs):
        """
        input_shape:  (tuple) tuple of the input dimension shape (channel, height, width)
        filters:       (list) list of the tuples consists of (number of channels, kernel size, and strides)
        use_batchnorm: (bool) whether to use batchnorm
        """ 
        super(ConvModel, self).__init__()   
        
        # Conv layers
        input_channel, input_height, input_width = input_shape
        filters = [(input_channel, 0, 0)] + filters
        output_height, output_width = input_height, input_width
        layers = []
        for i in range(len(filters)-1):
            in_channels  = filters[i][0]
            out_channels = filters[i+1][0]
            kernel_size  = filters[i+1][1]
            stride       = filters[i+1][2]
            layers.append(conv_block(in_channels, out_channels, kernel_size, stride, use_batchnorm))

            output_height = conv2d_size_out(output_height, kernel_size, stride)
            output_width = conv2d_size_out(output_width, kernel_size, stride)

        self.layers = nn.Sequential(*layers)
        self.output_shape = (out_channels, output_height, output_width)
        self.output_dim = out_channels * output_height * output_width
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    
class ConvToMlpModel(nn.Module):
    """
    Model object 
    """
    
    def __init__(self,
                 input_shape = (3, 84, 84), 
                 filters = [(32, 8, 4), (64, 4, 2), (64, 3, 1)],
                 hidden_dims = [512],
                 use_batchnorm = False,
                 **kwargs):
        """
        input_shape:  (tuple) tuple of the input dimension shape (channel, height, width)
        filters:       (list) list of the tuples consists of (number of channels, kernel size, and strides)
        hidden_dims:   (list) list of the dimensions for the hidden layers  
        use_batchnorm: (bool) whether to use batchnorm
        """ 
        super(ConvToMlpModel, self).__init__()   
        self.conv = ConvModel(input_shape, filters, use_batchnorm)
        self.mlp  = MlpModel(self.conv.output_dim, hidden_dims, use_batchnorm)
        self.output_dim = hidden_dims[-1]
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x