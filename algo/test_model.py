import os
import numpy as np
import torch
from sklearn import metrics
from data_mod.format_data import DataFormatter
from data_mod.viz_image import VizImage
from data_mod.data_analysis import DataAnalyse
from algo.nn_models import BinPixNN, DistPixNN, RingPixNN
from algo.tree_forest import DecisionTree, RandomForest
from joblib import load
import utils


class Modelvaler:
    """Class to val models"""

    def __init__(self):
        self.data_dir = utils.load_config("PATH", "DATA_DIR")
        self.device = torch.device(utils.load_config("TRAINING_INFO", "DEVICE"))
        self.data_formatter = DataFormatter()
        self.val_leaves = utils.load_config("DATA", "VALIDATION_LEAVES")
        self.visualise = VizImage()
        self.threshold = utils.load_config(
            "TRAINING_INFO", "LAB_MASK", "MLP", "LABEL_THRESHOLD"
        )
        self.data_type = utils.load_config("TRAINING_CHOICE", "DATA_TYPE")
        self.model_type = utils.load_config("TRAINING_CHOICE", "MODEL_TYPE")

    def performance(self, y_val, y_predicted):
        if self.data_type == "lab_mask":
            return self.performance_2class(y_val, y_predicted)
        if self.data_type == "dist_mask":
            return self.performance_continuous(y_val, y_predicted)
        if self.data_type == "ring_mask":
            return self.performance_ring(y_val, y_predicted)

    def performance_continuous(self, y_val, y_predicted):
        mse = metrics.mean_squared_error(y_val, y_predicted)
        print("Model's performances on val dataset: ")
        print(f"Mean squared error = {mse:.2f}")
        metrics_dictionary = {"mse": mse}
        return metrics_dictionary

    def performance_ring(self, y_val, y_pred):
        target_names = ["healthy", "ring", "sick"]
        print(
            "Classification Report:\n",
            metrics.classification_report(y_val, y_pred),
            # target_names=target_names,
        )
        class_dict = metrics.classification_report(
            y_val,
            y_pred,
            output_dict=True,  # target_names=target_names
        )
        return class_dict["0.0"]

    def performance_2class(self, y_val, y_predicted):
        """Print performance information of a 2-class classification model

        Returns
        -------
        metrics_dictionary : dict
            keys = accuracy, recall, precision, f1_score"""
        y_predicted = np.where(y_predicted <= self.threshold, 0, 1)
        accuracy = metrics.accuracy_score(y_true=y_val, y_pred=y_predicted)
        recall = metrics.recall_score(y_true=y_val, y_pred=y_predicted)
        precision = metrics.precision_score(y_true=y_val, y_pred=y_predicted)
        f1 = metrics.f1_score(y_true=y_val, y_pred=y_predicted)

        print("Model's performances on val dataset: ")
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
        return metrics_dictionary

    def open_val_data(self):
        original_X_val, y_set = self.data_formatter.load_data(
            leaf_numbers=self.val_leaves
        )
        X_val, y_val = self.data_formatter.scale_and_split_data(
            np.copy(original_X_val), y_set
        )
        return X_val, y_val, original_X_val

    def load_nn_and_perf(self, model_path, model_name, X_val, y_val):
        """Load model, print performance, and returns y_pred"""
        if self.data_type == "lab_mask":
            loaded_model = BinPixNN().to(self.device)
        if self.data_type == "dist_mask":
            loaded_model = DistPixNN().to(self.device)
        if self.data_type == "ring_mask":
            loaded_model = RingPixNN().to(self.device)
        loaded_model.load_state_dict(torch.load(model_path))
        loaded_model.eval()
        print(f"performance of model {model_name} on val dataset :")
        with torch.no_grad():
            # Print model performance
            y_predicted = loaded_model(X_val)
            y_val = y_val.to("cpu").numpy().flatten()
            y_predicted = y_predicted.to("cpu").numpy()
            if self.data_type == "ring_mask":
                # to 0 (healthy), 1 (ring), 2 (sick)
                y_val[y_val == 255] = 0
                y_val[y_val == 100] = 1
                y_val[y_val == 200] = 2

                def keep_likely_class(x):
                    return np.argmax(x)

                # To 1D
                y_predicted = np.apply_along_axis(
                    func1d=keep_likely_class, axis=1, arr=y_predicted
                )
                """Flatten pose un problème pour ring mask puisque les labels sont de dimension 3. Il faut réfléchir à comment je deal avec ça.
                Pour le moment je choisis la catégorie avec la plus grande probabilité."""
            else:
                y_predicted = y_predicted.flatten()
            # Copy to not round the y_predicted about to be returned
            self.performance(y_val, np.copy(y_predicted))
        return y_predicted

    def tree_perf(self):
        """Prints performance of all the saved decision tree models"""
        models_dir = os.path.join(self.data_dir, "..", "model_backup", "tree")
        model_names = os.listdir(models_dir)
        for model_name in model_names:
            channels = utils.get_channels_from_name(model_name)
            x_set, y_set = self.data_formatter.load_data(
                channels=channels, leaf_numbers=self.val_leaves
            )
            X_val, y_val = self.data_formatter.scale_and_split_data(
                x_set, y_set, to_tensor=False, scale=False
            )
            self.one_tree_perf(model_name, X_val, y_val)

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
            X_val, y_val = self.data_formatter.leaf_mask_data(leaf)
            X_val, y_val = self.data_formatter.scale_and_split_data(X_val, y_val)
            y_pred = self.load_nn_and_perf(model_path, model_name, X_val, y_val)
            if round and self.data_type == "lab_mask":
                y_pred = np.where(y_pred <= self.threshold, 0, 1)
            y_leaf, y_pred = self.data_formatter.reconstitute_leaf(leaf, y_pred)
            self.visualise.plot_y_real_pred(
                y_leaf, y_pred, title=leaf + ", model : MLP"
            )

        if model_extension == "joblib":
            channels = utils.load_config(
                "TRAINING_INFO",
                self.data_type.upper(),
                self.model_type.upper(),
                "CHANNELS",
            )
            X_val, y_val = self.data_formatter.leaf_mask_data(leaf)
            X_val, y_val = self.data_formatter.scale_and_split_data(
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

    def compare_class_spectra(self, model_path):
        """Opens data and calls data_analysis method `plot_spectra`,
        displaying channel intensity distribution for each class (TP, TN, FP, FN)"""
        model_name = model_path.split("/")[-1]
        X_val, y_val, X_raw = self.open_val_data(model_name)
        y_pred = self.load_nn_and_perf(model_path, model_name, X_val, y_val)
        y_val = y_val.to("cpu").numpy().flatten()
        y_val = y_val.astype(bool)
        y_pred = np.where(y_pred <= self.threshold, 0, 1).astype(bool)
        TN = X_raw[~y_pred & ~y_val]
        TP = X_raw[y_pred & y_val]
        FP = X_raw[y_pred & ~y_val]
        FN = X_raw[~y_pred & y_val]
        data_analyser = DataAnalyse()
        data_analyser.plot_spectra([TN, TP, FP, FN], ["TN", "TP", "FP", "FN"])


if __name__ == "__main__":
    model_valer = Modelvaler()
    # MODELS_DIR = os.path.join(
    #     model_valer.data_dir, "..", "model_backup", "neural_network"
    # )
    # model_valer.nn_perf(models_dir=MODELS_DIR)
    # model_valer.tree_perf()

    LEAF = "foliolo2_enves_a9"
    MODEL_PATH_MLP = "/home/colind/work/Mines/TR_DIMA/DIMA_code/model_backup/nn_binary/2026-03-23,16:47_MLP-on-lab_mask_1000epochs_lr:0.3_111features_balanced:False_.pth"
    MODEL_PATH_TREE = "/home/colind/work/Mines/TR_DIMA/DIMA_code/data/../model_backup/tree/2026-03-31,10:18_tree_max-depth:4_channels:[64,68,65]_balanced:False_.joblib"
    # model_valer.compare_class_spectra(MODEL_PATH_MLP)
    MODEL_PATH_FOREST = "/home/colind/work/Mines/TR_DIMA/DIMA_code/model_backup/rd_forest/2026-03-27,16:57_rdforest_nestimators:100_balanced:False_.joblib"
    MODEL_PATH_MLP = "/home/colind/work/Mines/TR_DIMA/DIMA_code/data/../model_backup/nn_binary/2026-03-30,13:49_MLP-on-lab_mask_1000epochs_lr:0.3_10features_balanced:False_.pth"
    MODEL_PATH_MLP = "/home/colind/work/Mines/TR_DIMA/DIMA_code/data/../model_backup/nn_ring/2026-03-31,09:41_MLP-on-ring_mask_1000epochs_lr:0.3_10features_balanced:False_.pth"
    model_valer.analyse_one_leaf(LEAF, MODEL_PATH_TREE, round=False)
