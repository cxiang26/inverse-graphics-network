import torch
from torch import nn

class GPNN(nn.Module):
    def __init__(self,
                 n_in,
                 n_hidden,
                 n_out):
        super(GPNN, self).__init__()

        ## L1 norm for hiddenLayers
        self.h1 = nn.Linear(in_features=n_in, out_features=n_hidden)
        self.activation = nn.ReLU()
        # self.h2 = nn.Linear(in_features=n_hidden, out_features=n_hidden)
        # # self.dp = nn.Dropout(.3)
        # self.h3 = nn.Linear(in_features=n_hidden, out_features=n_hidden)

        ## L2 norm for outputLayer
        self.outputLayer = nn.Linear(in_features=n_hidden, out_features=n_out)

    def forward(self, x):
        x1 = self.h1(x)
        x1 = self.activation(x1)
        # x2 = self.h2(x1)
        # x2 = self.activation(x2)
        #
        # x3 = self.h3(x2)
        # x3 = self.activation(x3)

        xo = self.outputLayer(x1 )
        return xo