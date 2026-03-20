import numpy as np
import os
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from sklearnex import (
    patch_sklearn,
)  # it should enable GPU for sklearn but doesn't seem to work

from open_image import OpenImage
from format_data import DataFormatter
from models import BinPixNN, DecisionTree, DistPixNN
from test_model import ModelTester
import utils
import train_utils


class Train:
    """
    Main class for data loading and model training
    """

    def __init__(self, model_type="MLP", data_type="lab_mask"):
        """
        - instanciates OpenImage and DataFormatter
        - set device to GPU, as attributes
        - set all useful info from CONFIG as attribute

        :param str model_type: by default, "MLP". Other possibilities are "CNN"

        """
        self.number_of_channels = (
            utils.load_config("DATA", "NUMBER_OF_CHANNELS")
            if type(utils.load_config("DATA", "NUMBER_OF_CHANNELS")) == int
            else utils.load_config("DATA", "TOTAL_N_CHANNELS")
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = model_type
        self.data_type = data_type

        self.open_im = OpenImage(number_of_channels=self.number_of_channels)
        self.data_formatter = DataFormatter(
            device=self.device,
            number_of_channels=self.number_of_channels,
            data_type=data_type,
        )
        self.model_tester = ModelTester(data_type=data_type)

        if torch.cuda.is_available():
            print("The GPU is available and will be used for computation.")
        else:
            print("The GPU is NOT available.")

        self.data_dir = utils.load_config("PATH", "DATA_DIR")
        self.tb_path = os.path.join(self.data_dir, "..", "runs")
        self.balance = utils.load_config("DATA", "BALANCE")
        self.test_leaves = utils.load_config("DATA", "TEST_LEAVES")
        self.validation_leaves = utils.load_config("DATA", "VALIDATION_LEAVES")
        self.train_leave_numbers = utils.leaf_training_list(
            self.test_leaves + self.validation_leaves
        )

        training_info = utils.load_config(
            "TRAINING_INFO", data_type.upper(), model_type.upper()
        )
        # Some models don't need a learning rate
        if "LEARNING_RATE" in training_info:
            self.learning_rate = training_info["LEARNING_RATE"]
        if "NUM_EPOCHS" in training_info:
            self.num_epochs = training_info["NUM_EPOCHS"]

        # for decision tree
        if "MAX_DEPTH" in training_info:
            self.max_depth = training_info["MAX_DEPTH"]
        if "CHANNELS" in training_info:
            self.tree_channels = training_info["CHANNELS"]

    def define_mlp_bin_functions(self):
        training_info = utils.load_config("TRAINING_INFO", "LAB_MASK", "MLP")
        self.model = BinPixNN(self.number_of_channels).to(self.device)
        # self.criterion = nn.BCELoss()
        self.criterion = train_utils.FocalLoss(alpha=1, gamma=2) 
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.step_lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            "min",
            factor=training_info["FACTOR"],
            patience=training_info["PATIENCE"],
            threshold=training_info["THRESHOLD"],
        )
        self.early_stopping = train_utils.EarlyStopping(patience=10)

    def define_mlp_dist_functions(self):
        training_info = utils.load_config("TRAINING_INFO", "DIST_MASK", "MLP")
        self.model = DistPixNN(self.number_of_channels).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.step_lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            "min",
            factor=training_info["FACTOR"],
            patience=training_info["PATIENCE"],
            threshold=training_info["THRESHOLD"],
        )
        self.early_stopping = train_utils.EarlyStopping(patience=100)

    def define_nn_functions(self):
        """Sets model, criterion, optimizer, lr_scheduler as attributes"""
        if self.model_type == "MLP" and self.data_type == "lab_mask":
            self.define_mlp_bin_functions()
        if self.model_type == "MLP" and self.data_type == "dist_mask":
            self.define_mlp_dist_functions()

        # step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.97)

    def loop_initialiser(self):
        """Returns all useful variables to start training loop, and define training functions

        Returns
        -------
        X_train, y_train, X_val, y_val, date
        """
        self.date = datetime.today().strftime("%Y-%m-%d,%H:%M")
        exp_path = os.path.join(
            self.tb_path, f"{self.model_type}-{self.data_type}", self.date
        )
        os.makedirs(exp_path, exist_ok=True)
        self.writer = SummaryWriter(exp_path)
        X_train, y_train = self.data_formatter.load_data(
            leaf_numbers=self.train_leave_numbers
        )
        X_train, y_train = self.data_formatter.scale_and_split_data(
            X_train, y_train, requires_grad=True
        )
        X_val, y_val = self.data_formatter.load_data(
            leaf_numbers=self.validation_leaves
        )
        X_val, y_val = self.data_formatter.scale_and_split_data(X_val, y_val)

        self.define_nn_functions()

        print(
            f"train set shape = {X_train.shape}; validation set shape =  {X_val.shape}"
        )
        return (
            X_train,
            y_train,
            X_val,
            y_val,
        )

    def epoch_info(self, epoch, training_loss, val_loss):
        """Logs and prints useful information for epoch"""
        self.writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": training_loss, "Validation": val_loss},
            epoch + 1,
        )
        self.writer.add_scalar(
            "learning_rate",
            self.step_lr_scheduler.get_last_lr()[0],
            epoch,
        )
        if (epoch + 1) % (max(self.num_epochs // 30, 1)) == 0 or epoch == 0:
            print(
                f"epoch: {epoch+1}, training_loss = {training_loss:.4f}, val_loss = {val_loss:.4f}, lr = {self.step_lr_scheduler.get_last_lr()[0]:.4f}"
            )

    def one_epoch(self, X_train, y_train, X_val, y_val):
        # validation
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
        return training_loss, val_loss

    def loop_nobatch(self):
        """Main training loop. All data is loaded at once before the beginning of the loop."""
        (
            X_train,
            y_train,
            X_val,
            y_val,
        ) = self.loop_initialiser()

        for epoch in tqdm(range(self.num_epochs), desc="training", unit="epoch"):
            training_loss, val_loss = self.one_epoch(X_train, y_train, X_val, y_val)
            self.epoch_info(epoch, training_loss, val_loss)
            # Check early stopping
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print(
                    f"Early stopping triggered at epoch {epoch}. Last (val_loss, train_loss) = {val_loss, training_loss}"
                )
                break

        self.end_loop(training_loss, val_loss)

    def end_loop(self, training_loss, val_loss):
        print(
            f"Final training_loss = {training_loss:.4f}, val_loss = {val_loss:.4f}, last learning rate = {self.step_lr_scheduler.get_last_lr()[0]}"
        )
        self.model.save_nn(
            self.early_stopping.best_model_state,
            file_name=f"{self.date}_{self.model_type}-on-{self.data_type}_{self.num_epochs}epochs_lr:{self.learning_rate}_{self.number_of_channels}features_balanced:{self.balance}_.pth",
        )
        # self.model is not exactly the model we have saved, but it probably is a good approximation.
        self.nn_results(self.model)

    def nn_results(self, model):
        """Saves model performance to tensorboard and prints it"""
        with torch.no_grad():
            # writer.add_image('mnist_images', img_grid) (to add an image
            # self.writer.add_graph(
            #     model, ex_vect[0, :]
            # )  # Don't know what it does exactly
            x_set, y_set = self.data_formatter.load_data(leaf_numbers=self.test_leaves)
            X_test, y_test = self.data_formatter.scale_and_split_data(x_set, y_set)
            # Print model performance
            y_predicted = model(X_test)
            y_test = y_test.to("cpu").numpy().flatten()
            y_predicted = y_predicted.to("cpu").numpy().flatten()
            # PR curve makes sense only for 2 class classification problems
            if self.data_type == "lab_mask":
                self.writer.add_pr_curve("recall curve", y_test, y_predicted)

            metrics_dictionary = self.model_tester.performance(y_test, y_predicted)
            hparam_dict = {
                "number of epochs": self.num_epochs,
                "number of features": self.number_of_channels,
                "balance dataset": self.balance,
                "initial lr": self.learning_rate,
            }
            self.writer.add_text("h_param", str(hparam_dict))
            self.writer.add_text("metrics", str(metrics_dictionary))
            self.writer.add_hparams(
                hparam_dict=hparam_dict, metric_dict=metrics_dictionary
            )
            self.writer.close()

    def decision_tree(self):
        """decision tree training"""
        date = datetime.today().strftime("%Y-%m-%d,%H:%M")
        exp_path = os.path.join(self.tb_path, "tree", "tree_" + date)
        os.makedirs(exp_path, exist_ok=True)
        self.writer = SummaryWriter(exp_path)
        # set all channels for opening hsi
        self.data_formatter.open_im.number_of_channels = utils.load_config(
            "DATA", "TOTAL_N_CHANNELS"
        )
        patch_sklearn()
        x_set, y_set = self.data_formatter.load_data(
            channels=self.tree_channels, leaf_numbers=self.train_leave_numbers
        )
        X_train, y_train = self.data_formatter.scale_and_split_data(
            x_set, y_set, to_tensor=False, scale=False
        )
        # Create Decision Tree classifer object
        clf = DecisionTree(max_depth=self.max_depth, channels=self.tree_channels)
        # Train Decision Tree Classifer
        clf = clf.fit(X_train, y_train)
        file_name = f"{date}_tree_max-depth:{self.max_depth}_channels:{str(self.tree_channels).replace(" ", "")}_balanced:{self.balance}_.joblib"
        clf.save_tree(file_name)
        self.tree_results(clf)

    def tree_results(self, clf):
        # Predict the response for test dataset

        x_set, y_set = self.data_formatter.load_data(
            channels=self.tree_channels, leaf_numbers=self.test_leaves
        )
        X_test, y_test = self.data_formatter.scale_and_split_data(
            x_set, y_set, to_tensor=False, scale=False
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

        # clf.viz_decision_tree()


if __name__ == "__main__":

    DATA_TYPE = "lab_mask"
    MODEL_TYPE = "MLP"

    trainer = Train(data_type=DATA_TYPE, model_type=MODEL_TYPE)
    trainer.loop_nobatch()

    # trainer = Train(data_type="lab_mask", model_type="tree")
    # trainer.decision_tree()
