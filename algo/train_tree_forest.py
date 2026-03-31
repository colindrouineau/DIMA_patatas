import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report

from data_mod.open_image import OpenImage
from data_mod.format_data import DataFormatter
from algo.test_model import ModelTester
import utils
from algo.tree_forest import DecisionTree, RandomForest

class TrainTree:

    def __init__(self):
        """
        - instanciates OpenImage and DataFormatter
        - set all useful info from CONFIG as attribute
        """
        self.date = datetime.today().strftime("%Y-%m-%d,%H:%M")
        self.number_of_channels = utils.load_config("DATA", "NUMBER_OF_CHANNELS")
        self.data_type = utils.load_config("TRAINING_CHOICE", "DATA_TYPE")

        self.open_im = OpenImage()
        self.data_formatter = DataFormatter()
        self.model_tester = ModelTester()

        self.data_dir = utils.load_config("PATH", "DATA_DIR")
        self.tb_path = os.path.join(self.data_dir, "..", "runs")
        self.balance = utils.load_config("TRAINING_CHOICE", "BALANCE")
        self.test_leaves = utils.load_config("DATA", "TEST_LEAVES")
        self.validation_leaves = utils.load_config("DATA", "VALIDATION_LEAVES")
        self.train_leave_numbers = utils.leaf_training_list(
            self.test_leaves + self.validation_leaves
        )
        self.max_depth = utils.load_config(
            "TRAINING_INFO", self.data_type.upper(), "TREE", "MAX_DEPTH"
        )
        self.tree_channels = utils.load_config(
            "TRAINING_INFO", self.data_type.upper(), "TREE", "CHANNELS"
        )
        exp_path = os.path.join(self.tb_path, f"TREE-{self.data_type}", self.date)
        self.writer = SummaryWriter(exp_path)

    def decision_tree(self):
        """decision tree training"""
        exp_path = os.path.join(self.tb_path, "tree", "tree_" + self.date)
        os.makedirs(exp_path, exist_ok=True)

        x_set, y_set = self.data_formatter.load_data(
            channels=self.tree_channels, leaf_numbers=self.train_leave_numbers
        )
        X_train, y_train = self.data_formatter.scale_and_split_data(
            x_set, y_set, to_tensor=False, scale=False, normalise=False
        )
        # Create Decision Tree classifer object
        clf = DecisionTree(max_depth=self.max_depth, channels=self.tree_channels)
        # Train Decision Tree Classifer
        clf = clf.fit(X_train, y_train)
        file_name = f"{self.date}_tree_max-depth:{self.max_depth}_channels:{str(self.tree_channels).replace(" ", "")}_balanced:{self.balance}_.joblib"

        clf.save_tree(file_name)
        self.tree_results(clf)

    def tree_results(self, clf):
        # Predict the response for test dataset

        x_set, y_set = self.data_formatter.load_data(
            channels=self.tree_channels, leaf_numbers=self.test_leaves
        )
        X_test, y_test = self.data_formatter.scale_and_split_data(
            x_set, y_set, to_tensor=False, scale=False, normalise=False
        )
        y_pred = clf.predict(X_test)
        metrics_dictionary = self.model_tester.performance(y_test, y_pred)
        hparam_dict = {
            "max_depth": self.max_depth,
            "channels": self.tree_channels,
            "balance dataset": self.balance,
        }
        self.writer.add_text("h_param", str(hparam_dict))
        self.writer.add_text("metrics", str(metrics_dictionary))
        # lists are not supported for hparam_dict
        hparam_dict["channels"] = str(hparam_dict["channels"])
        self.writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metrics_dictionary)
        self.writer.close()


class TrainForest:

    def __init__(self):
        """
        - instanciates OpenImage and DataFormatter
        - set all useful info from CONFIG as attribute
        """
        self.date = datetime.today().strftime("%Y-%m-%d,%H:%M")
        self.data_type = utils.load_config("TRAINING_CHOICE", "DATA_TYPE")

        self.open_im = OpenImage()
        self.data_formatter = DataFormatter()
        self.model_tester = ModelTester()

        self.data_dir = utils.load_config("PATH", "DATA_DIR")
        self.tb_path = os.path.join(self.data_dir, "..", "runs")
        self.balance = utils.load_config("TRAINING_CHOICE", "BALANCE")
        self.test_leaves = utils.load_config("DATA", "TEST_LEAVES")
        self.validation_leaves = utils.load_config("DATA", "VALIDATION_LEAVES")
        self.train_leave_numbers = utils.leaf_training_list(
            self.test_leaves + self.validation_leaves
        )

        training_info = utils.load_config(
            "TRAINING_INFO", self.data_type.upper(), "RANDOM_FOREST"
        )
        self.forest_channels = training_info["CHANNELS"]
        self.n_estimators = training_info["N_ESTIMATORS"]

    def random_forest(self):
        x_set, y_set = self.data_formatter.load_data(
            channels=self.forest_channels, leaf_numbers=self.train_leave_numbers
        )
        X_train, y_train = self.data_formatter.scale_and_split_data(
            x_set, y_set, to_tensor=False, scale=False
        )
        rf_classifier = RandomForest(n_estimators=self.n_estimators)
        rf_classifier.fit(X_train, y_train)
        self.random_forest_results(rf_classifier)
        file_name = f"{self.date}_rdforest_nestimators:{self.n_estimators}__nchannels:{len(self.forest_channels)}_balanced:{self.balance}_.joblib"
        rf_classifier.save_forest(file_name)

    def random_forest_results(self, rf_classifier):
        x_set, y_set = self.data_formatter.load_data(
            channels=self.forest_channels, leaf_numbers=self.test_leaves
        )
        X_test, y_test = self.data_formatter.scale_and_split_data(
            x_set, y_set, to_tensor=False, scale=False
        )
        y_pred = rf_classifier.predict(X_test)
        classification_rep = classification_report(y_test, y_pred)

        print("Classification Report:\n", classification_rep)


if __name__ == "__main__":
    # forest = TrainForest()
    # forest.random_forest()  # launch training. Modify params is CONFIG file
    # tree = TrainTree()
    # tree.decision_tree()
    import joblib
    print(joblib.load("/home/colind/work/Mines/TR_DIMA/DIMA_code/data/../model_backup/tree/2026-03-31,10:18_tree_max-depth:4_channels:[64,68,65]_balanced:False_.joblib"))
