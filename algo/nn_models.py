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


class CommonNN(nn.Module):
    """Simple MLP for pixel health binary classification"""

    def __init__(self):
        super(CommonNN, self).__init__()
        input_size = len(utils.load_config("TRAINING_CHOICE", "CHANNELS"))
        hidden_size = input_size * 3
        # self.dropout = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.leaky_relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

    def save_nn(self, best_model_state, nn_trace, file_name):
        save_model(nn_trace, best_model_state, file_name, "lab_mask")
