import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report

from data_mod.open_image import OpenImage
from data_mod.format_data import DataFormatter
from algo.test_model import ModelTester
import utils
from algo.tree_forest import DecisionTree, RandomForest
from joblib import load


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
        self.validation_leaves = utils.load_config("DATA", "VALIDATION_LEAVES")
        self.train_leave_numbers = utils.leaf_training_list()
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
        X_train, y_train = self.data_formatter.scale_and_format_data(
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
        # Predict the response for validation dataset

        x_set, y_set = self.data_formatter.load_data(
            channels=self.tree_channels, leaf_numbers=self.validation_leaves
        )
        X_val, y_val = self.data_formatter.scale_and_format_data(
            x_set, y_set, to_tensor=False, scale=False, normalise=False
        )
        y_pred = clf.predict(X_val)
        metrics_dictionary = self.model_tester.performance(y_val, y_pred)
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
        self.validation_leaves = utils.load_config("DATA", "VALIDATION_LEAVES")
        self.train_leave_numbers = utils.leaf_training_list()

        training_info = utils.load_config(
            "TRAINING_INFO", self.data_type.upper(), "RANDOM_FOREST"
        )
        self.forest_channels = training_info["CHANNELS"]
        self.n_estimators = training_info["N_ESTIMATORS"]

    def random_forest(self):
        x_set, y_set = self.data_formatter.load_data(
            channels=self.forest_channels, leaf_numbers=self.train_leave_numbers
        )
        X_train, y_train = self.data_formatter.scale_and_format_data(
            x_set, y_set, to_tensor=False, scale=False
        )
        rf_classifier = RandomForest(n_estimators=self.n_estimators)
        rf_classifier.fit(X_train, y_train)
        self.random_forest_results(rf_classifier)
        file_name = f"{self.date}_rdforest_nestimators:{self.n_estimators}__nchannels:{len(self.forest_channels)}_balanced:{self.balance}_.joblib"
        rf_classifier.save_forest(file_name)

    def random_forest_results(self, rf_classifier):
        x_set, y_set = self.data_formatter.load_data(
            channels=self.forest_channels, leaf_numbers=self.validation_leaves
        )
        X_val, y_val = self.data_formatter.scale_and_format_data(
            x_set, y_set, to_tensor=False, scale=False
        )
        y_pred = rf_classifier.predict(X_val)
        classification_rep = classification_report(y_val, y_pred)

        print("Classification Report:\n", classification_rep)

    def tree_perf(self):
        """Prints performance of all the saved decision tree models"""
        models_dir = os.path.join(self.data_dir, "..", "model_backup", "tree")
        model_names = os.listdir(models_dir)
        for model_name in model_names:
            channels = utils.get_channels_from_name(model_name)
            x_set, y_set = self.data_formatter.load_data(
                channels=channels, leaf_numbers=self.val_leaves
            )
            X_val, y_val = self.data_formatter.scale_and_format_data(
                x_set, y_set, to_tensor=False, scale=False
            )
            self.one_tree_perf(model_name, X_val, y_val)


class Tester:
    def one_tree_perf(self, model_name, X_val, y_val):
        models_dir = os.path.join(self.data_dir, "..", "model_backup", "tree")
        model_path = os.path.join(models_dir, model_name)
        # Load the saved model
        loaded_model_joblib = load(model_path)
        try:
            print(f"performance of model {model_name} :")
            y_predicted = loaded_model_joblib.predict(X_val)
            y_val = y_val.flatten()
            y_predicted = y_predicted.flatten()
            self.performance_2class(y_val, y_predicted)
            print()
        except Exception as e:
            print(e)
        finally:
            return y_predicted

    def one_forest_perf(self, model_name, X_val, y_val):
        models_dir = os.path.join(self.data_dir, "..", "model_backup", "rd_forest")
        model_path = os.path.join(models_dir, model_name)
        # Load the saved model
        loaded_model_joblib = load(model_path)
        try:
            print(f"performance of model {model_name} :")
            y_pred = loaded_model_joblib.predict(X_val)
            y_val = y_val.flatten()
            y_pred = y_pred.flatten()
            print(
                "Classification Report:\n",
                metrics.classification_report(y_val, y_pred),
            )
        except Exception as e:
            print(e)
        finally:
            return y_pred

    if model_extension == "joblib":
        channels = utils.load_config(
            "TRAINING_INFO",
            self.data_type.upper(),
            self.model_type.upper(),
            "CHANNELS",
        )
        X_val, y_val = self.data_formatter.leaf_mask_data(leaf)
        X_val, y_val = self.data_formatter.scale_and_format_data(
            X_val, y_val, scale=False, to_tensor=False
        )
        X_val = X_val[:, channels]
        if self.model_type == "TREE":
            y_pred = self.one_tree_perf(model_name, X_val, y_val)
            y_leaf, y_pred = self.data_formatter.reconstitute_leaf(leaf, y_pred)
            self.visualise.plot_y_real_pred(
                y_leaf, y_pred, title=leaf + ", model : tree"
            )
        if self.model_type == "RANDOM_FOREST":
            y_pred = self.one_forest_perf(model_name, X_val, y_val)
            y_leaf, y_pred = self.data_formatter.reconstitute_leaf(leaf, y_pred)
            self.visualise.plot_y_real_pred(
                y_leaf, y_pred, title=leaf + ", model : random forest"
            )


if __name__ == "__main__":
    # forest = TrainForest()
    # forest.random_forest()  # launch training. Modify params is CONFIG file
    # tree = TrainTree()
    # tree.decision_tree()
    import joblib

    print(
        joblib.load(
            "/home/colind/work/Mines/TR_DIMA/DIMA_code/data/../model_backup/tree/2026-03-31,10:18_tree_max-depth:4_channels:[64,68,65]_balanced:False_.joblib"
        )
    )
