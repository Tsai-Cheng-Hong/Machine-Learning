import torch.nn as nn
import torch

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.nn_layers = nn.Sequential(
            # fully connected layer
            # Can stack number of layers you want
            # Note that the first layer's in_features need to match to data's dim.
            # And out_features need to match to label's dim
            nn.Linear(in_features=223, out_features=512),  #fisrt layer
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=512, out_features=512),  #second layer
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=512, out_features=1024), #third layer
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(in_features=1024, out_features=20),
        )

    def forward(self, x):
        # data fit into model, no need to rewrite
        x = self.nn_layers(x)
        return x


