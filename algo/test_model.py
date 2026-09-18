import sys
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
import torch


from torch import jit

from sklearn import metrics
from data_mod.format_data import DataFormatter
from data_mod.viz_image import VizImage, COLORS
from data_mod.data_analysis import DataAnalyse
from algo.nn_models import (
    CommonNN,
)
import utils


class ModelTester:
    """Class to val models"""

    def __init__(self, model_path, round_labels=False, threshold=0.9, real_test=False):
        self.data_dir = utils.load_config("PATH", "DATA_DIR")
        self.device = torch.device(utils.load_config("TRAINING_INFO", "DEVICE"))
        self.data_formatter = DataFormatter(test=True, balance_data=False)
        val_leaves = utils.load_config("DATA", "VALIDATION_LEAVES")
        test_leaves = utils.load_config("DATA", "TEST_LEAVES")
        self.leaves = test_leaves if real_test else val_leaves
        self.visualise = VizImage()
        self.data_type = utils.load_config("TRAINING_CHOICE", "DATA_TYPE")
        self.model_type = utils.load_config("TRAINING_CHOICE", "MODEL_TYPE")
        self.round = round_labels
        self.model_path = model_path
        if model_path is not None:
            self.model_name = model_path.split("/")[-1]
        self.threshold = threshold
        self.real_test = real_test

    def performance_on_whole_dataset(self, thresh_search=False):
        """Prints performance of model on the whole validation dataset"""
        with torch.no_grad():
            # writer.add_image('mnist_images', img_grid) (to add an image
            X_val, y_val, _ = self.open_val_data()
            utils.cprint(
                f"Performance of model {self.model_name} on validation dataset:"
            )
            y_pred, y_val = self.load_nn_and_perf(X_val, y_val)
            if thresh_search and not self.real_test:
                self.threshold, _ = self.find_best_threshold(y_val, y_pred, show=True)

    def find_best_threshold(self, y_val, y_pred, show=True):
        if len(np.unique(y_val)) > 2:
            return
        n = 100
        thresholds = np.linspace(0, 1, n)

        # Calculate F1 score for each threshold
        f1_scores = []
        for threshold in thresholds:
            f1 = metrics.f1_score(y_val, (y_pred >= threshold).astype(int))
            f1_scores.append(f1)

        best_threshold = np.argmax(f1_scores) / n
        best_score = np.max(f1_scores)
        utils.cprint(
            f"best_threshold is {best_threshold:.4f} with f1_score = {best_score:.4f}",
            colour="RED",
        )

        if show:
            plt.figure(figsize=(8, 6))
            plt.plot(thresholds, f1_scores, label="F1 Score", color="blue")
            plt.xlabel("Threshold")
            plt.ylabel("F1 Score")
            plt.title("F1 Score as a Function of Threshold")
            plt.grid(True)
            plt.legend()
            plt.show()
        return best_threshold, best_score

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

    def open_val_data(self):
        original_X_val, y_set = self.data_formatter.load_data(leaf_numbers=self.leaves)
        X_val, y_val = self.data_formatter.scale_and_format_data(
            np.copy(original_X_val), y_set
        )
        return X_val, y_val, original_X_val

    def load_model(self, model_path):
        if "whole_model_backup" in model_path:  # then we load the whole mode
            loaded_model = jit.load(model_path).to(self.device)
        else:
            loaded_model = CommonNN().to(self.device)
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
                print(
                    f"Make sure you selected the right number of channels for the loaded model. (number_of_channels = {len(self.channels)})"
                )
                sys.exit()
            y_val = y_val.to("cpu").numpy()
            y_predicted = y_predicted.to("cpu").numpy()
            _, y_predicted, y_val = self.performance_2class(y_val, y_predicted)
        return y_predicted, y_val

    def analyse_one_leaf(self, leaf):
        """
        Shows predicted label distribution, and gives performance for the specific leaf

        :param str leaf: leaf_name
        """
        utils.cprint(f"Performance of model {self.model_name} on leaf {leaf} :")

        X_val, y_val = self.data_formatter.leaf_mask_data(leaf)
        X_val, y_val = self.data_formatter.scale_and_format_data(X_val, y_val)
        y_pred, y_val = self.load_nn_and_perf(X_val, y_val)
        y_leaf, y_pred = self.data_formatter.reconstitute_leaf(leaf, y_pred)
        self.visualise.plot_y_real_pred(
            y_leaf,
            y_pred,
            title=f"Leaf {leaf}, data_type = {self.data_type}, model = {self.model_name}",
        )

    def error_type_on_leaf(self, leaf):
        """
        Plots the leaf with coloured pixels for each class (TN, FN, TP, FP). Possibility to click on a pixel to visualise its spectrogram.
        """
        channels = utils.load_config("TRAINING_CHOICE", "CHANNELS")
        utils.cprint(f"Performance of model {self.model_name} on leaf {leaf} :")

        round_save = self.round
        self.round = True

        X_val, y_val = self.data_formatter.leaf_mask_data(leaf)
        X_val, y_val = self.data_formatter.scale_and_format_data(X_val, y_val)
        y_pred, y_val = self.load_nn_and_perf(X_val, y_val)
        y_leaf, y_pred = self.data_formatter.reconstitute_leaf(leaf, y_pred)
        X_val = X_val.to("cpu").numpy()
        _, X_val = self.data_formatter.reconstitute_leaf(leaf, X_val)

        height, width = y_leaf.shape
        y_class = np.zeros((height, width, 3), dtype=np.uint8)
        y_class[y_leaf == 0] = (255, 255, 255)
        y_class[(y_leaf == 255) & (y_pred == 0)] = (0, 180, 0)  # TN
        y_class[(y_leaf == 200) & (y_pred == 0)] = (255, 0, 0)  # FN
        y_class[(y_leaf == 255) & (y_pred == 1)] = (255, 255, 0)  # FP
        y_class[(y_leaf == 200) & (y_pred == 1)] = (0, 0, 0)  # TP

        legend_elements = [
            Patch(facecolor=(0 / 255, 180 / 255, 0 / 255), label="TN"),
            Patch(facecolor=(255 / 255, 0 / 255, 0 / 255), label="FN"),
            Patch(facecolor=(255 / 255, 255 / 255, 0 / 255), label="FP"),
            Patch(facecolor=(0, 0, 0), label="TP"),
        ]

        # Create figure and subplots
        fig, (ax_image, ax_spectrum) = plt.subplots(1, 2, figsize=(10, 5))
        plt.subplots_adjust(wspace=0.4)

        # Display the selected channel
        im = ax_image.imshow(y_class)
        ax_image.legend(handles=legend_elements, loc="upper right")
        ax_image.set_title(f"Prediction class on {leaf}")

        # Initialize the spectrum subplot
        (line,) = ax_spectrum.plot([], [])
        ax_spectrum.set_title("Pixel Spectrum")
        y_lim = (0, 1)
        ax_spectrum.set_ylim(y_lim)
        ax_spectrum.set_xlabel("channel")
        ax_spectrum.set_ylabel("intensity")

        # Store all spectra and their corresponding lines
        spectra_lines = []
        spectra_data = []
        crosses = []

        # Function to handle mouse clicks
        def on_click(event):
            nonlocal spectra_lines, spectra_data, crosses
            if event.inaxes != ax_image:
                return  # Ignore clicks outside the image subplot

                # Right click: Reset the spectrum subplot
            if event.button == 3:  # Right mouse button
                for line in spectra_lines:
                    line.remove()
                for cross in crosses:
                    for part in cross:
                        part.remove()
                spectra_lines = []
                spectra_data = []
                crosses = []
                ax_spectrum.set_title("Pixel Spectrum")
                fig.canvas.draw()
                return

            # Get the clicked pixel coordinates (rounded to nearest integer)
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            
            # Ensure the click is within the image bounds
            if 0 <= x < X_val.shape[1] and 0 <= y < X_val.shape[0]:
                spectrum = X_val[y, x, :]
                spectra_data.append((x, y, spectrum))
                color = COLORS[len(spectra_lines) % 10]
                # Plot the new spectrum
                (line,) = ax_spectrum.plot(
                    channels,
                    spectrum,
                    color=color,
                    label=f"Pixel ({x}, {y})",
                )
                spectra_lines.append(line)

                # Draw a cross on the image at (x, y) with the same color
                cross_horizontal = ax_image.plot(
                    [x - 3, x + 3], [y, y], color=color, linewidth=1.5
                )
                cross_vertical = ax_image.plot(
                    [x, x], [y - 3, y + 3], color=color, linewidth=1.5
                )
                crosses.append([cross_horizontal[0], cross_vertical[0]])

                ax_spectrum.legend()
                ax_spectrum.relim()
                ax_spectrum.autoscale_view()
                ax_spectrum.set_title(f"Pixel Spectra (Last: {x}, {y})")
                fig.canvas.draw()

        # Connect the click event
        fig.canvas.mpl_connect("button_press_event", on_click)
        plt.show()


        self.round = round_save

    def compare_class_spectra(self):
        """Opens data and calls data_analysis method `plot_spectra`,
        displaying channel intensity distribution for each class (TP, TN, FP, FN)

        NOTE : works only for 2 class classification."""
        self.round = True
        X_val, y_val, X_raw = self.open_val_data()
        utils.cprint(f"Performance of model {self.model_name} on validation dataset :")
        y_pred, y_val = self.load_nn_and_perf(X_val, y_val)
        y_pred = np.where(y_pred <= self.threshold, 0, 1).astype(bool)
        TN = X_raw[~y_pred & ~y_val]
        TP = X_raw[y_pred & y_val]
        FP = X_raw[y_pred & ~y_val]
        FN = X_raw[~y_pred & y_val]
        data_analyser = DataAnalyse()
        data_analyser.plot_spectra([TN, TP, FP, FN], ["TN", "TP", "FP", "FN"])


if __name__ == "__main__":
    MODEL_PATH_MLP = "/home/colind/work/Mines/TR_DIMA/DIMA_code/data/../model_info/model_backup/lab_mask/09-09--15:50_MLP.pth"

    model_tester = ModelTester(
        model_path=MODEL_PATH_MLP, round_labels=False, real_test=False, threshold=0.7
    )

    LEAF = "foliolo1_enves_a9"

    # model_tester.performance_on_whole_dataset(thresh_search=True)
    # model_tester.analyse_one_leaf(LEAF)
    # model_tester.compare_class_spectra()

    model_tester.error_type_on_leaf(LEAF)

    for i in range(10):
        LEAF = "foliolo12_enves_a" + str(5 + i)
        # model_tester.analyse_one_leaf(LEAF)
