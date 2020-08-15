import torch
from torch import nn
from torch.nn import functional as F
from poolings import *
from CNNs import *
from loss import *

class SpeakerClassifier(nn.Module):

    def __init__(self, parameters, device):
        super().__init__()
        
        parameters.feature_size = 80 

        self.pooling_method = parameters.pooling_method
        self.device = device
        
        if parameters.front_end=='VGG3L':
            self.vector_size = getVGG3LOutputDimension(parameters.feature_size, outputChannel=parameters.kernel_size)
            self.front_end = VGG3L(parameters.kernel_size)
        
        if parameters.front_end=='VGG4L':
            self.vector_size = getVGG4LOutputDimension(parameters.feature_size, outputChannel=parameters.kernel_size)
            self.front_end = VGG4L(parameters.kernel_size)
        
        self.pooling_method = parameters.pooling_method

        if parameters.pooling_method == 'Statistical':
            self.PoolingLayer = StatisticalPooling()
            self.vector_size *= 2
        elif parameters.pooling_method == 'Attention':
            self.PoolingLayer = Attention(self.vector_size)
        elif parameters.pooling_method == 'MHA':
            self.PoolingLayer = MultiHeadAttention(self.vector_size, parameters.heads_number)
        elif parameters.pooling_method == 'DoubleMHA':
            self.PoolingLayer = DoubleMHA(self.vector_size, parameters.heads_number, mask_prob = parameters.mask_prob)
            self.vector_size = self.vector_size//parameters.heads_number
        
        self.fc1 = nn.Linear(self.vector_size, parameters.embedding_size)
        self.b1 = nn.BatchNorm1d(parameters.embedding_size)
        self.fc2 = nn.Linear(parameters.embedding_size, parameters.embedding_size)
        self.b2 = nn.BatchNorm1d(parameters.embedding_size)
        self.preLayer = nn.Linear(parameters.embedding_size, parameters.embedding_size)
        self.b3 = nn.BatchNorm1d(parameters.embedding_size)
        
        if parameters.loss == 'Softmax':
            self.predictionLayer = nn.Linear(parameters.embedding_size, parameters.num_spkrs)
        elif parameters.loss == 'AMSoftmax':
            self.predictionLayer = AMSoftmax(parameters.embedding_size, parameters.num_spkrs, s=parameters.scalingFactor, m = parameters.marginFactor)

        self.loss = parameters.loss
    
    def getEmbedding(self,x):

        encoder_output = self.front_end(x)

        embedding0, alignment = self.PoolingLayer(encoder_output)
        embedding1 = F.relu(self.fc1(embedding0))
        embedding2 = self.b2(F.relu(self.fc2(embedding1)))
    
        return encoder_output, embedding2, None 


    def forward(self, x, label=None):

        encoder_output = self.front_end(x)

        embedding0, alignment = self.PoolingLayer(encoder_output)
        embedding1 = F.relu(self.fc1(embedding0))
        embedding2 = self.b2(F.relu(self.fc2(embedding1)))
                
        if self.loss == 'Softmax':
            embedding3 = self.b3(F.relu(self.preLayer(embedding2)))
            prediction = self.predictionLayer(embedding3)

        elif self.loss == 'AMSoftmax':
            embedding3 = self.preLayer(embedding2)
            prediction = self.predictionLayer(embedding3, label)
        
        return encoder_output, embedding2, prediction

