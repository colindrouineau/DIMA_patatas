import os
import torch.nn as nn
import torch

import utils


class NeuralNetCommon:
    """shared neural net functions"""

    def __init__(self):
        pass

    @classmethod  # later probably should make it an object method
    def save_nn(cls, model, file_name, state_dict, mode):
        # save only state dict
        nn_backup_path = os.path.join(
            utils.load_config("PATH", "DATA_DIR"), "..", "model_backup", "nn_" + mode
        )
        os.makedirs(nn_backup_path, exist_ok=True)
        # Potentially make "MLP" a var
        file = os.path.join(
            nn_backup_path,
            file_name,
        )
        torch.save(state_dict, file)
        print(f"Model dict saved as {file}")

        nn_whole_model_backup_path = os.path.join(
            utils.load_config("PATH", "DATA_DIR"),
            "..",
            "whole_model_backup",
            "nn_" + mode,
        )
        os.makedirs(nn_whole_model_backup_path, exist_ok=True)
        # Potentially make "MLP" a var
        file = os.path.join(
            nn_whole_model_backup_path,
            file_name,
        )
        torch.save(model, file)

        # print(model.state_dict())
        print(f"Model saved as {file}")


class BinPixNN(nn.Module):
    """Simple MLP for pixel health binary classification"""

    def __init__(self):
        super(BinPixNN, self).__init__()
        input_size = utils.load_config("DATA", "NUMBER_OF_CHANNELS")
        hidden_size = utils.load_config(
            "TRAINING_INFO", "LAB_MASK", "MLP", "HIDDEN_SIZE"
        )
        self.dropout = nn.Dropout(p=0.2)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.dropout(x)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out

    def save_nn(self, best_model_state, file_name):
        """Saves model state_dict. best_model_state is an attribute of EarlyStopping class instance."""
        NeuralNetCommon.save_nn(self, file_name, best_model_state, mode="binary")


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

    def save_nn(self, best_model_state, file_name):
        """Saves model state_dict"""
        NeuralNetCommon.save_nn(file_name, best_model_state, mode="distance")


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
        self.softmax = nn.Softmax()

    def forward(self, x):
        out = self.dropout(x)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.softmax(out)
        return out

    def save_nn(self, best_model_state, file_name):
        """Saves model state_dict. best_model_state is an attribute of EarlyStopping class instance."""
        NeuralNetCommon.save_nn(self, file_name, best_model_state, mode="ring")
