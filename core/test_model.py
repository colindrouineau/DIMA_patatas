import os
import numpy as np
import torch
from sklearn import metrics
from format_data import DataFormatter
from model import NeuralNet
from joblib import load
import utils


class ModelTester:
    """Class to test models"""

    def __init__(self):
        self.data_dir = utils.load_config("PATH", "DATA_DIR")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_formatter = DataFormatter(
            device=self.device,
            number_of_channels=utils.load_config("DATA", "TOTAL_N_CHANNELS"),
        )
        self.test_leaves = utils.load_config("DATA", "TEST_LEAVES")

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
        y_predicted = y_predicted.round()
        accuracy = metrics.accuracy_score(y_true=y_test, y_pred=y_predicted)
        recall = metrics.recall_score(y_true=y_test, y_pred=y_predicted)
        precision = metrics.precision_score(y_true=y_test, y_pred=y_predicted)
        f1 = metrics.f1_score(y_true=y_test, y_pred=y_predicted)

        print("Model's performances : ")
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

    def nn_perf(self):
        """Prints performance of all the saved nn models"""
        models_dir = os.path.join(self.data_dir, "..", "model_backup", "neural_network")
        model_names = os.listdir(models_dir)
        for model_name in model_names:
            number_of_channels = utils.get_nfeatures_from_name(model_name)
            # To load the right channels :
            self.data_formatter.open_im.number_of_channels = number_of_channels
            self.data_formatter.number_of_channels = number_of_channels
            x_set, y_set = self.data_formatter.load_data(leaf_numbers=self.test_leaves)
            X_test, y_test = self.data_formatter.scale_and_split_data(
                x_set, y_set
            )
            model_path = os.path.join(models_dir, model_name)
            loaded_model = NeuralNet(input_size=number_of_channels).to(self.device)
            try:
                loaded_model.load_state_dict(torch.load(model_path))
                loaded_model.eval()
                print(f"performance of model {model_name} :")
                with torch.no_grad():
                    # Print model performance
                    y_predicted = loaded_model(X_test)
                    y_test = y_test.to("cpu").numpy().flatten()
                    y_predicted = y_predicted.to("cpu").numpy().flatten()
                    self.performance_2class(y_test, y_predicted)
                print()
            except Exception as e:
                print(e)

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


if __name__ == "__main__":
    model_tester = ModelTester()
    model_tester.nn_perf()
    model_tester.tree_perf()
