import torch.nn as nn
import utils

class NeuralNet(nn.Module):
    """Simple MLP"""
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        hidden_size = utils.load_config("TRAINING_INFO", "HIDDEN_SIZE")
        if hidden_size == "HALF":
            hidden_size = input_size // 2
        if hidden_size == "INPUT":
            hidden_size = input_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.sigmoid(out)
        return out
    

