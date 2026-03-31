import os
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from six import StringIO
import pydotplus
from joblib import dump  # for tree saving
from sklearn.ensemble import RandomForestClassifier

import utils

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

class RandomForest(RandomForestClassifier):
    def __init__(self, n_estimators):
        super(RandomForest, self).__init__(n_estimators=n_estimators, random_state=42, n_jobs=-1, verbose=1)

    def save_forest(self, file_name):
        tree_backup_path = os.path.join(
            utils.load_config("PATH", "DATA_DIR"),
            "..",
            "model_backup",
            "rd_forest",
        )
        os.makedirs(tree_backup_path, exist_ok=True)
        file = os.path.join(
            tree_backup_path,
            file_name,
        )
        dump(self, file)  # Serialize the model to disk
        print(f"Model saved as {file}")

