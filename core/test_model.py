import os
import numpy as np
import torch
from sklearn import metrics
from format_data import DataFormatter
from viz_image import VizImage
from models import BinPixNN, DistPixNN
from joblib import load
import utils


class ModelTester:
    """Class to test models"""

    def __init__(self, data_type="lab_mask"):
        self.data_dir = utils.load_config("PATH", "DATA_DIR")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_formatter = DataFormatter(
            device=self.device,
            number_of_channels=utils.load_config("DATA", "TOTAL_N_CHANNELS"),
            data_type=data_type,
        )
        self.test_leaves = utils.load_config("DATA", "TEST_LEAVES")
        self.visualise = VizImage(
            number_of_channels=utils.load_config("DATA", "TOTAL_N_CHANNELS")
        )
        self.threshold = utils.load_config(
            "TRAINING_INFO", "LAB_MASK", "MLP", "LABEL_THRESHOLD"
        )
        self.data_type = data_type

    def performance(self, y_test, y_predicted):
        if self.data_type == "lab_mask":
            return self.performance_2class(y_test, y_predicted)
        if self.data_type == "dist_mask":
            return self.performance_continuous(y_test, y_predicted)

    def performance_continuous(self, y_test, y_predicted):
        mse = metrics.mean_squared_error(y_test, y_predicted)
        print("Model's performances on test dataset: ")
        print(f"Mean squared error = {mse:.2f}")
        metrics_dictionary = {"MSE": mse}
        return metrics_dictionary

    def performance_2class(
        self,
        y_test,
        y_predicted,
    ):
        """Print performance information of a 2-class classification model

        Returns
        -------
        metrics_dictionary : dict
            keys = accuracy, recall, precision, f1_score"""
        y_predicted = np.where(y_predicted <= self.threshold, 0, 1)
        accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_predicted)
        recall = metrics.recall_score(y_true=y_test, y_pred=y_predicted)
        precision = metrics.precision_score(y_true=y_test, y_pred=y_predicted)
        f1 = metrics.f1_score(y_true=y_test, y_pred=y_predicted)

        print("Model's performances on test dataset: ")
        print(f"- accuracy: {100 * accuracy:.2f} %")
        print(f"- recall: {100 * recall:.2f} %")
        print(f"- precision: {100 * precision:.2f} %")
        print(f"- f1_score: {100 * f1:.2f} %")

        metrics_dictionary = {
            "accuracy": round(accuracy, 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "F1_score": round(f1, 4),
        }
        return metrics_dictionary

    def nn_perf(self, models_dir):
        """Prints performance of all the saved nn models in a directory"""
        model_names = os.listdir(models_dir)
        for model_name in model_names:
            if model_name.split(".")[-1] != "pth":
                print(f"{model_name} is not a valid model name")
                continue
            number_of_channels = utils.get_nfeatures_from_name(model_name)
            # To load the right channels :
            self.data_formatter.open_im.number_of_channels = number_of_channels
            self.data_formatter.number_of_channels = number_of_channels
            x_set, y_set = self.data_formatter.load_data(leaf_numbers=self.test_leaves)
            X_test, y_test = self.data_formatter.scale_and_split_data(x_set, y_set)
            model_path = os.path.join(models_dir, model_name)
            self.load_nn_and_perf(
                number_of_channels, model_path, model_name, X_test, y_test
            )

    def load_nn_and_perf(
        self, number_of_channels, model_path, model_name, X_test, y_test
    ):
        """Load model, print performance, and returns y_pred"""
        if self.data_type == "lab_mask":
            loaded_model = BinPixNN(input_size=number_of_channels).to(self.device)
        if self.data_type == "dist_mask":
            loaded_model = DistPixNN(input_size=number_of_channels).to(self.device)
        try:
            loaded_model.load_state_dict(torch.load(model_path))
            loaded_model.eval()
            print(f"performance of model {model_name} on test dataset :")
            with torch.no_grad():
                # Print model performance
                y_predicted = loaded_model(X_test)
                y_test = y_test.to("cpu").numpy().flatten()
                y_predicted = y_predicted.to("cpu").numpy().flatten()
                # Copy to not round the y_predicted about to be returned
                self.performance(y_test, np.copy(y_predicted))
            print()
        except Exception as e:
            print(e)
            y_predicted = None
        finally:
            return y_predicted

    def tree_perf(self):
        """Prints performance of all the saved decision tree models"""
        models_dir = os.path.join(self.data_dir, "..", "model_backup", "tree")
        model_names = os.listdir(models_dir)
        for model_name in model_names:
            channels = utils.get_channels_from_name(model_name)
            x_set, y_set = self.data_formatter.load_data(
                channels=channels, leaf_numbers=self.test_leaves
            )
            X_test, y_test = self.data_formatter.scale_and_split_data(
                x_set, y_set, to_tensor=False, scale=False
            )
            self.one_tree_perf(model_name, X_test, y_test)

    def one_tree_perf(self, model_name, X_test, y_test):
        models_dir = os.path.join(self.data_dir, "..", "model_backup", "tree")
        model_path = os.path.join(models_dir, model_name)
        # Load the saved model
        loaded_model_joblib = load(model_path)
        try:
            print(f"performance of model {model_name} :")
            y_predicted = loaded_model_joblib.predict(X_test)
            y_test = y_test.flatten()
            y_predicted = y_predicted.flatten()
            self.performance_2class(y_test, y_predicted)
            print()
        except Exception as e:
            print(e)
        finally:
            return y_predicted

    def analyse_one_leaf(self, leaf, model_path, round=False):
        """
        Shows predicted label distribution, and gives performance for the specific leaf

        :param str leaf: leaf_name
        :param str model_path: path where the model was saved
        :param bool round: if True, y_predicted values will be 0 or 1
        """
        model_extension = model_path.split(".")[-1]
        model_name = model_path.split("/")[-1]

        if model_extension == "pth":
            number_of_channels = utils.get_nfeatures_from_name(model_name)
            self.data_formatter.open_im.number_of_channels = number_of_channels
            self.data_formatter.number_of_channels = number_of_channels
            X_test, y_test = self.data_formatter.leaf_mask_data(leaf)
            X_test, y_test = self.data_formatter.scale_and_split_data(X_test, y_test)
            y_pred = self.load_nn_and_perf(
                number_of_channels, model_path, model_name, X_test, y_test
            )
            if round and self.data_type == "lab_mask":
                y_pred = np.where(y_pred <= self.threshold, 0, 1)
            y_leaf, y_pred = self.data_formatter.reconstitute_leaf(leaf, y_pred)
            self.visualise.plot_y_real_pred(
                y_leaf, y_pred, title=leaf + ", model : MLP"
            )

        if model_extension == "joblib":
            channels = utils.get_channels_from_name(model_name)
            X_test, y_test = self.data_formatter.leaf_mask_data(leaf)
            X_test, y_test = self.data_formatter.scale_and_split_data(
                X_test, y_test, scale=False, to_tensor=False
            )
            X_test = X_test[:, channels]
            y_pred = self.one_tree_perf(model_name, X_test, y_test)
            y_leaf, y_pred = self.data_formatter.reconstitute_leaf(leaf, y_pred)
            self.visualise.plot_y_real_pred(
                y_leaf, y_pred, title=leaf + ", model : tree"
            )


if __name__ == "__main__":
    DATA_TYPE = "lab_mask"
    model_tester = ModelTester(data_type=DATA_TYPE)
    # MODELS_DIR = os.path.join(
    #     model_tester.data_dir, "..", "model_backup", "neural_network"
    # )
    # model_tester.nn_perf(models_dir=MODELS_DIR)
    # model_tester.tree_perf()

    LEAF = "foliolo7_enves_a10"
    MODEL_PATH_MLP = "/home/colind/work/Mines/TR_DIMA/DIMA_code/data/../model_backup/nn_binary/2026-03-20,23:17_MLP-on-lab_mask_1000epochs_lr:0.3_15features_balanced:False_.pth"
    MODEL_PATH_TREE = "/home/colind/work/Mines/TR_DIMA/DIMA_code/data/../model_backup/tree/2026-03-20,10:44_tree_max-depth:4_channels:[64,68,65]_balanced:False_.joblib"
    model_tester.analyse_one_leaf(LEAF, MODEL_PATH_MLP, round=False)
