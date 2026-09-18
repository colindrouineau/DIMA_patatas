import numpy as np
import os
from tqdm import tqdm
from datetime import datetime
import sys
import shutil

import torch
from torch import jit
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn import metrics
import matplotlib.pyplot as plt

import mlflow
import mlflow.pytorch
from mlflow.types import TensorSpec, Schema
from mlflow.models import ModelSignature
from torchviz import make_dot

from data_mod.open_image import OpenImage
from data_mod.format_data import DataFormatter
from algo.nn_models import CommonNN
from algo.test_model import ModelTester
import utils
import algo.train_utils as train_utils


class TrainNN:
    """
    Main class for data loading and model training
    """

    def __init__(self):
        """
        - Instantiates OpenImage and DataFormatter
        - Sets device to GPU as attributes
        - Sets all useful info from CONFIG as attributes
        """
        self.date = datetime.today().strftime("%d-%m--%H:%M")
        self.channels = utils.load_config("TRAINING_CHOICE", "CHANNELS")
        self.model_type = utils.load_config("TRAINING_CHOICE", "MODEL_TYPE")
        self.data_type = utils.load_config("TRAINING_CHOICE", "DATA_TYPE")

        self.open_im = OpenImage()
        self.data_formatter = DataFormatter()

        if torch.cuda.is_available():
            print("The GPU is available and will be used for computation.")
        else:
            print("The GPU is NOT available.")

        self.data_dir = utils.load_config("PATH", "DATA_DIR")
        self.exp_name = self.date + "_" + self.model_type

        # Set MLflow experiment
        mlflow.set_experiment(self.exp_name)

        self.validation_leaves = utils.load_config("DATA", "VALIDATION_LEAVES")
        self.train_leave_numbers = utils.leaf_training_list()
        self.device = torch.device(utils.load_config("TRAINING_INFO", "DEVICE"))
        training_info = utils.load_config("TRAINING_INFO", self.model_type.upper())
        self.learning_rate = training_info["LEARNING_RATE"]
        self.num_epochs = training_info["NUM_EPOCHS"]
        self.threshold = training_info["LABEL_THRESHOLD"]

        self.model_tester = ModelTester(
            model_path=None, threshold=self.threshold
        )  # We don't need to load a model, just performance method

    def define_mlp_bin_functions(self):
        training_info = utils.load_config("TRAINING_INFO", "MLP")
        self.model = CommonNN().to(self.device)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.step_lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            "min",
            factor=training_info["FACTOR"],
            patience=training_info["PATIENCE_LR"],
            threshold=training_info["DELTA"],
        )
        self.early_stopping = train_utils.EarlyStopping(
            patience=training_info["PATIENCE_STOP"], delta=training_info["DELTA"]
        )

    def define_nn_functions(self):
        """Sets model, criterion, optimizer, lr_scheduler as attributes"""
        self.define_mlp_bin_functions()

    def loop_initialiser(self):
        """Returns all useful variables to start training loop, and define training functions

        Returns
        -------
        X_train, y_train, X_val, y_val, date
        """
        X_train, y_train = self.data_formatter.load_data(
            leaf_numbers=self.train_leave_numbers
        )
        X_train, y_train = self.data_formatter.scale_and_format_data(
            X_train, y_train, requires_grad=True
        )
        X_val, y_val = self.data_formatter.load_data(
            leaf_numbers=self.validation_leaves
        )
        X_val, y_val = self.data_formatter.scale_and_format_data(X_val, y_val)

        self.define_nn_functions()

        print(
            f"train set shape = {X_train.shape}; validation set shape =  {X_val.shape}"
        )
        return X_train, y_train, X_val, y_val

    def epoch_info(self, epoch, epoch_result):
        """Logs and prints useful information for epoch"""
        training_loss, val_loss, (y_train, y_pred), (y_val, y_pred_val) = epoch_result

        # Log metrics to MLflow
        mlflow.log_metric("Training loss", training_loss, step=epoch + 1)
        mlflow.log_metric("Validation loss", val_loss, step=epoch + 1)
        mlflow.log_metric(
            "Learning_rate", self.step_lr_scheduler.get_last_lr()[0], step=epoch + 1
        )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                y_pred_round = np.where(
                    y_pred.to("cpu").numpy() <= self.threshold, 0, 1
                )
                y_pred_val_round = np.where(
                    y_pred_val.to("cpu").numpy() <= self.threshold, 0, 1
                )
                f1_score_training = metrics.f1_score(
                    y_true=y_train.to("cpu"), y_pred=y_pred_round
                )
                f1_score_val = metrics.f1_score(
                    y_true=y_val.to("cpu"), y_pred=y_pred_val_round
                )

            mlflow.log_metric("F1 score training", f1_score_training, step=epoch + 1)
            mlflow.log_metric("F1 score validation", f1_score_val, step=epoch + 1)

            print(
                f"epoch: {epoch+1}, training_loss = {training_loss:.4f}, val_loss = {val_loss:.4f}, lr = {self.step_lr_scheduler.get_last_lr()[0]:.4f}"
            )
            print(
                f"F1 training data = {f1_score_training:.4f}, F1 validation data = {f1_score_val:.4f}"
            )

    def one_epoch(self, X_train, y_train, X_val, y_val):
        # Validation
        self.model.eval()
        with torch.no_grad():
            y_pred_val = self.model(X_val)
            val_loss = self.criterion(y_pred_val, y_val).item()

        # Train
        self.model.train(True)
        y_pred = self.model(X_train)
        loss = self.criterion(y_pred, y_train)
        training_loss = loss.item()

        loss.backward()
        self.step_lr_scheduler.step(val_loss)
        self.optimizer.step()
        self.optimizer.zero_grad()

        assert not bool(np.isnan(training_loss)) and not bool(
            np.isnan(val_loss)
        ), f"val_loss or training_loss became undefined (vloss = {val_loss}, tloss = {training_loss})"

        return training_loss, val_loss, (y_train, y_pred), (y_val, y_pred_val)

    def main_loop(self):
        """Main training loop. All data is loaded at once before the beginning of the loop."""
        X_train, y_train, X_val, y_val = self.loop_initialiser()

        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(
                {
                    "model_type": self.model_type,
                    "data_type": self.data_type,
                    "learning_rate": self.learning_rate,
                    "num_epochs": self.num_epochs,
                    "threshold": self.threshold,
                    "channels": self.channels,
                }
            )

            try:
                for epoch in tqdm(
                    range(self.num_epochs), desc="training", unit="epoch"
                ):
                    epoch_result = self.one_epoch(X_train, y_train, X_val, y_val)
                    self.epoch_info(epoch, epoch_result)
                    training_loss, val_loss = epoch_result[0:2]

                    # Check early stopping
                    self.early_stopping(val_loss, self.model)
                    if self.early_stopping.early_stop:
                        print(
                            f"Early stopping triggered at epoch {epoch}. Last (val_loss, train_loss) = {round(val_loss, 4), round(training_loss, 4)}"
                        )
                        break
            except KeyboardInterrupt:
                print(
                    f"Training was interrupted by user. The model will be saved in its last state."
                )

            self.end_loop(training_loss, val_loss)

    def end_loop(self, training_loss, val_loss):
        print(
            f"Final training_loss = {training_loss:.4f}, val_loss = {val_loss:.4f}, last learning rate = {self.step_lr_scheduler.get_last_lr()[0]}"
        )
        self.model.load_state_dict(self.early_stopping.best_model_state)
        self.model.eval()
        self.nn_results()

        # To save the best model found
        self.early_stopping.load_best_model(self.model)

    def nn_results(self):
        """Saves model performance to MLflow and prints it"""
        with torch.no_grad():
            x_set, y_set = self.data_formatter.load_data(
                leaf_numbers=self.validation_leaves
            )
            X_val, y_val = self.data_formatter.scale_and_format_data(x_set, y_set)

            # Log model graph as an artifact (optional)
            # Note: MLflow doesn't directly support logging model graphs like TensorBoard,
            # but you can save it as an artifact.
            model_graph = os.path.join("model_graph.png")
            # Generate the graph
            graph = make_dot(
                self.model(X_val), params=dict(self.model.named_parameters())
            )
            graph.render("model_graph", format="png", cleanup=True)
            mlflow.log_artifact(model_graph)

            # Trace the model
            self.nn_trace = jit.trace(self.model, X_val)

            # Print model performance
            y_predicted = self.model(X_val)
            y_val = y_val.to("cpu").numpy()
            y_predicted = y_predicted.to("cpu").numpy()
            print(f"Performance of model {self.exp_name} on validation dataset")

            metrics_dictionary, y_predicted, y_val = (
                self.model_tester.performance_2class(y_val, y_predicted)
            )

            # Log metrics dictionary
            for metric_name, metric_value in metrics_dictionary.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log PR curve as an artifact
            plt.figure()
            metrics.PrecisionRecallDisplay.from_predictions(y_val, y_predicted).plot()
            pr_curve_path = "pr_curve.png"
            plt.savefig(pr_curve_path)
            mlflow.log_artifact(pr_curve_path)
            plt.close()

            input_signature = Schema([
                TensorSpec(
                    np.dtype("float32"),
                    (-1, X_val.shape[1]),
                    name="input",
                )
            ])

            signature = ModelSignature(inputs=input_signature)

            mlflow.pytorch.log_model(
                self.model,
                name="model",
                serialization_format="pt2",
                signature=signature,
                input_example=X_val
)

            # Log hyperparameters and training info
            hparam_dict = {
                "number of epochs": self.num_epochs,
                "number of features": len(self.channels),
                "initial lr": self.learning_rate,
            }
            mlflow.log_params(hparam_dict)

            training_info = utils.load_config("TRAINING_INFO", self.model_type.upper())
            training_info = str(training_info)
            training_functions = f"\n\nModel is: {self.model}, \n Loss function is: {self.criterion}, \n Optimizer is {self.optimizer}"

            # Log training info as a text file
            with open("training_info.txt", "w") as f:
                f.write(training_info + training_functions)
            mlflow.log_artifact("training_info.txt")

            save = input("Do you want to save this model? (Y/n) ")
            if save not in ["", "y", "Y"]:
                log_dir = os.path.join(
                    self.data_dir,
                    "..",
                    "model_info",
                    "runs",
                    self.data_type,
                    self.exp_name,
                )
                confirm = input(
                    f"You are about to forget this model and delete the folder '{log_dir}' and all the files and folders it contains. Are you sure? (type 'rm' to delete)"
                )
                if confirm == "rm":
                    shutil.rmtree(log_dir)
                    sys.exit()


if __name__ == "__main__":
    # To choose training type, change CONFIG file.
    trainer = TrainNN()
    trainer.main_loop()
