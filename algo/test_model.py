import sys
import numpy as np
import torch
  

from torch import jit

from sklearn import metrics
from data_mod.format_data import DataFormatter
from data_mod.viz_image import VizImage
from data_mod.data_analysis import DataAnalyse
from algo.nn_models import BinPixNN, DistPixNN, RingPixNN
import utils


class ModelTester:
    """Class to val models"""

    def __init__(self, model_path, round_labels=False):
        self.data_dir = utils.load_config("PATH", "DATA_DIR")
        self.device = torch.device(utils.load_config("TRAINING_INFO", "DEVICE"))
        self.data_formatter = DataFormatter()
        self.data_formatter.balance = False  # Balance is only useful for training
        self.val_leaves = utils.load_config("DATA", "VALIDATION_LEAVES")
        self.visualise = VizImage()
        self.threshold = utils.load_config(
            "TRAINING_INFO", "LAB_MASK", "MLP", "LABEL_THRESHOLD"
        )
        self.data_type = utils.load_config("TRAINING_CHOICE", "DATA_TYPE")
        self.model_type = utils.load_config("TRAINING_CHOICE", "MODEL_TYPE")
        self.round = round_labels
        self.model_path = model_path
        if model_path is not None:
            self.model_name = model_path.split("/")[-1]

    def performance_on_whole_val_set(self):
        """Prints performance of model on the whole validation dataset"""
        with torch.no_grad():
            # writer.add_image('mnist_images', img_grid) (to add an image
            x_set, y_set = self.data_formatter.load_data(leaf_numbers=self.val_leaves)
            X_val, y_val = self.data_formatter.scale_and_format_data(x_set, y_set)
            print(f"Performance of model {self.model_name} on validation dataset :")
            self.load_nn_and_perf(X_val, y_val)

    def performance(self, y_val, y_predicted):
        """Prints model performance, returns metrics, and formats the labels to match `analyse_one_leaf` requirements
        
        Returns
        -------
        metrics_dictionary, y_predicted, y_val
        """
        if self.data_type == "lab_mask":
            return self.performance_2class(y_val, y_predicted)
        if self.data_type == "dist_mask":
            return self.performance_continuous(y_val, y_predicted)
        if self.data_type == "ring_mask":
            return self.performance_ring(y_val, y_predicted)

    def performance_2class(self, y_val, y_predicted) -> tuple:
        """Print performance information of a 2-class classification model

        Returns
        -------
        metrics_dictionary : dict
            keys = accuracy, recall, precision, f1_score
        y_predicted 
        y_val"""
        y_predicted = y_predicted.flatten()
        y_val = y_val.flatten()
        y_pred = np.where(y_predicted <= self.threshold, 0, 1).astype(bool)
        y_valid = np.where(y_val <= self.threshold, 0, 1).astype(bool)
        if self.round:  # change value of predicted labels
            y_predicted = y_pred
            y_val = y_valid
        accuracy = metrics.accuracy_score(y_true=y_val, y_pred=y_pred)
        recall = metrics.recall_score(y_true=y_val, y_pred=y_pred)
        precision = metrics.precision_score(y_true=y_val, y_pred=y_pred)
        f1 = metrics.f1_score(y_true=y_val, y_pred=y_pred)

        print(f"- accuracy: {100 * accuracy:.2f} %")
        print(f"- recall: {100 * recall:.2f} %")
        print(f"- precision: {100 * precision:.2f} %")
        print(f"- f1_score: {100 * f1:.2f} %")

        metrics_dictionary = {
            "accuracy": round(accuracy, 4),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
            "f1_score": round(f1, 4),
        }
        return metrics_dictionary, y_predicted, y_val

    def performance_continuous(self, y_val, y_predicted):
        y_val = y_val.flatten()
        y_predicted = y_predicted.flatten()
        mse = metrics.mean_squared_error(y_val, y_predicted)
        print("Model's performances on val dataset: ")
        print(f"Mean squared error = {mse:.2f}")
        metrics_dictionary = {"mse": mse}
        return metrics_dictionary, y_predicted, y_val

    def performance_ring(self, y_val, y_pred):
        # to 0 (healthy), 1 (ring), 2 (sick)
        y_val[y_val == 255] = 0
        y_val[y_val == 100] = 1
        y_val[y_val == 200] = 2

        def keep_likely_class(x):
            return np.argmax(x)

        # To 1D
        y_pred = np.apply_along_axis(func1d=keep_likely_class, axis=1, arr=y_pred)
        y_val = np.apply_along_axis(func1d=keep_likely_class, axis=1, arr=y_val)
        """Flatten pose un problème pour ring mask puisque les labels sont de dimension 3.
        Il faut réfléchir à comment je deal avec ça.
        Pour le moment je choisis la catégorie avec la plus grande probabilité."""
        target_names = ["healthy", "ring", "sick"]
        print(
            "Classification Report:\n",
            metrics.classification_report(y_val, y_pred, target_names=target_names),
        )
        class_dict = metrics.classification_report(
            y_val, y_pred, output_dict=True, target_names=target_names, zero_division="warn"
        )
        return class_dict["ring"], y_pred, y_val

    def open_val_data(self):
        original_X_val, y_set = self.data_formatter.load_data(
            leaf_numbers=self.val_leaves
        )
        X_val, y_val = self.data_formatter.scale_and_format_data(
            np.copy(original_X_val), y_set
        )
        return X_val, y_val, original_X_val



    def load_model(self, model_path):
        if "whole_model_backup" in model_path:  # then we load the whole mode
            loaded_model = jit.load(model_path).to(self.device)
        else:
            if self.data_type == "lab_mask":
                loaded_model = BinPixNN().to(self.device)
            if self.data_type == "dist_mask":
                loaded_model = DistPixNN().to(self.device)
            if self.data_type == "ring_mask":
                loaded_model = RingPixNN().to(self.device)
            loaded_model.load_state_dict(torch.load(model_path))
        loaded_model.eval()
        return loaded_model

    def load_nn_and_perf(self, X_val, y_val) -> tuple:
        """Load model, print performance, and returns y_pred

        Returns
        -------
        y_predicted, y_val
            after transformation through performance function"""
        loaded_model = self.load_model(self.model_path)
        with torch.no_grad():
            # Print model performance
            try:
                y_predicted = loaded_model(X_val)
            except Exception as e:
                print(e)
                print(f"Make sure you selected the right number of channels for the loaded model. (number_of_channels = {self.data_formatter.number_of_channels})")
                sys.exit()
            y_val = y_val.to("cpu").numpy()
            y_predicted = y_predicted.to("cpu").numpy()
            _, y_predicted, y_val = self.performance(y_val, y_predicted)
        return y_predicted, y_val

    def analyse_one_leaf(self, leaf):
        """
        Shows predicted label distribution, and gives performance for the specific leaf

        :param str leaf: leaf_name
        :param str model_path: path where the model was saved
        """
        print(f"Performance of model {self.model_name} on leaf {leaf} :")

        X_val, y_val = self.data_formatter.leaf_mask_data(leaf)
        X_val, y_val = self.data_formatter.scale_and_format_data(X_val, y_val)
        y_pred, y_val = self.load_nn_and_perf(X_val, y_val)
        y_leaf, y_pred = self.data_formatter.reconstitute_leaf(leaf, y_pred)
        self.visualise.plot_y_real_pred(
            y_leaf,
            y_pred,
            title=f"Leaf {leaf}, data_type = {self.data_type}, model = {self.model_name}",
        )

    def compare_class_spectra(self):
        """Opens data and calls data_analysis method `plot_spectra`,
        displaying channel intensity distribution for each class (TP, TN, FP, FN)

        Note : works only for 2 class classification."""
        self.round = True
        X_val, y_val, X_raw = self.open_val_data()
        print(f"Performance of model {self.model_name} on validation dataset :")
        y_pred, y_val = self.load_nn_and_perf(X_val, y_val)
        TN = X_raw[~y_pred & ~y_val]
        TP = X_raw[y_pred & y_val]
        FP = X_raw[y_pred & ~y_val]
        FN = X_raw[~y_pred & y_val]
        data_analyser = DataAnalyse()
        data_analyser.plot_spectra([TN, TP, FP, FN], ["TN", "TP", "FP", "FN"])


if __name__ == "__main__":
    MODEL_PATH_MLP = "/home/colind/work/Mines/TR_DIMA/DIMA_code/data/../whole_model_backup/lab_mask/02-04--16:22_MLP.zip"
    model_tester = ModelTester(model_path=MODEL_PATH_MLP, round_labels=True)

    LEAF = "foliolo2_enves_a9"

    model_tester.performance_on_whole_val_set()
    model_tester.analyse_one_leaf(LEAF)
    model_tester.compare_class_spectra()
