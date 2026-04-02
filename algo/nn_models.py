import os
import torch.nn as nn
import torch
from torch import jit

import utils


def save_model(trace, best_model_state, file_name, data_type):
    """Saves model state_dict. best_model_state is an attribute of EarlyStopping class instance."""
    nn_backup_path = os.path.join(
        utils.load_config("PATH", "DATA_DIR"), "..", "model_backup", data_type
    )
    os.makedirs(nn_backup_path, exist_ok=True)
    file = os.path.join(nn_backup_path, file_name)
    torch.save(best_model_state, file)
    print(f"Model dict saved as {file}")

    # also save the whole model
    nn_backup_path = os.path.join(
        utils.load_config("PATH", "DATA_DIR"), "..", "whole_model_backup", data_type
    )
    os.makedirs(nn_backup_path, exist_ok=True)
    file_name = file_name.split(".")[0] + ".zip"
    file = os.path.join(nn_backup_path, file_name)
    jit.save(trace, file)
    print(f"Model saved as {file}")


class BinPixNN(nn.Module):
    """Simple MLP for pixel health binary classification"""

    def __init__(self):
        super(BinPixNN, self).__init__()
        input_size = utils.load_config("DATA", "NUMBER_OF_CHANNELS")
        hidden_size = (
            utils.load_config("TRAINING_INFO", "LAB_MASK", "MLP", "HIDDEN_SIZE")
        )
        self.dropout = nn.Dropout(p=0.1)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.dropout(x)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.sigmoid(out)
        return out

    def save_nn(self, best_model_state, nn_trace, file_name):
        save_model(nn_trace, best_model_state, file_name, "lab_mask")


class DistPixNN(nn.Module):
    """Simple MLP for pixel health binary classification"""

    def __init__(
        self,
    ):
        super(DistPixNN, self).__init__()
        input_size = utils.load_config("DATA", "NUMBER_OF_CHANNELS")
        hidden_size = utils.load_config(
            "TRAINING_INFO", "LAB_MASK", "MLP", "HIDDEN_SIZE"
        )
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 40)
        self.linear3 = nn.Linear(40, hidden_size)
        self.linear4 = nn.Linear(hidden_size, hidden_size)
        self.linear5 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out)
        out = self.linear3(out)
        out = self.relu(out)
        out = self.linear4(out)
        out = self.relu(out)
        out = self.linear5(out)
        out = self.relu(out)
        return out

    def save_nn(self, best_model_state, nn_trace, file_name):
        """Saves model state_dict. best_model_state is an attribute of EarlyStopping class instance."""
        save_model(nn_trace, best_model_state, file_name, "dist_mask")


class RingPixNN(nn.Module):
    """Simple MLP for pixel health binary classification"""

    def __init__(self):
        super(RingPixNN, self).__init__()
        input_size = utils.load_config("DATA", "NUMBER_OF_CHANNELS")
        hidden_size = utils.load_config(
            "TRAINING_INFO", "RING_MASK", "MLP", "HIDDEN_SIZE"
        )
        self.dropout = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 3)
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        out = self.dropout(x)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.softmax(out)
        return out

    def save_nn(self, best_model_state, nn_trace, file_name):
        save_model(nn_trace, best_model_state, file_name, "ring_mask")
