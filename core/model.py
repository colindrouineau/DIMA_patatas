import os
import torch.nn as nn
import torch
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from six import StringIO
import pydotplus
from joblib import dump  # for tree saving

import utils


class NeuralNet(nn.Module):
    """Simple MLP"""

    def __init__(self, input_size: int):
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

    def save_nn(self, file_name):
        """Saves model state_dict"""
        # save only state dict
        nn_backup_path = os.path.join(
            utils.load_config("PATH", "DATA_DIR"),
            "..",
            "model_backup",
            "neural_network",
        )
        os.makedirs(nn_backup_path, exist_ok=True)
        # Potentially make "MLP" a var
        file = os.path.join(
            nn_backup_path,
            file_name,
        )
        torch.save(self.state_dict(), file)
        # print(model.state_dict())
        print(f"Model saved as {file}")


class DecisionTree(DecisionTreeClassifier):
    def __init__(self, max_depth, channels):
        super(DecisionTree, self).__init__(max_depth=max_depth)
        self.channels = channels

    def save_tree(self, file_name):
        tree_backup_path = os.path.join(
            utils.load_config("PATH", "DATA_DIR"),
            "..",
            "model_backup",
            "tree",
        )
        os.makedirs(tree_backup_path, exist_ok=True)
        file = os.path.join(
            tree_backup_path,
            file_name,
        )
        dump(self, file)  # Serialize the model to disk
        print(f"Model saved as {file}")

    def viz_decision_tree(self):
        dot_data = StringIO()
        export_graphviz(
            self,
            out_file=dot_data,
            filled=True,
            rounded=True,
            special_characters=True,
            feature_names=[str(channel) for channel in self.channels],
            class_names=["0", "1"],
        )
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        tree_folder_path = os.path.join(
            utils.load_config("PATH", "DATA_DIR"), "..", "viz", "tree_graphs"
        )
        os.makedirs((tree_folder_path), exist_ok=True)
        graph.write_png(os.path.join(tree_folder_path, "tree.png"))
