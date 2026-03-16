import numpy as np
import os
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from sklearnex import (
    patch_sklearn,
)  # it should enable GPU for sklearn but doesn't seem to work

from open_image import OpenImage
from format_data import DataFormatter
from model import NeuralNet, DecisionTree
from test_model import ModelTester
import utils


class Train:
    """
    Main class for data loading and model training
    """

    def __init__(self):
        """
        - instanciates OpenImage and DataFormatter
        - set device to GPU, as attributes
        - set all useful info from CONFIG as attribute
        """
        self.number_of_channels = (
            utils.load_config("DATA", "NUMBER_OF_CHANNELS")
            if type(utils.load_config("DATA", "NUMBER_OF_CHANNELS")) == int
            else utils.load_config("DATA", "TOTAL_N_CHANNELS")
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.open_im = OpenImage(number_of_channels=self.number_of_channels)
        self.data_formatter = DataFormatter(
            device=self.device, number_of_channels=self.number_of_channels
        )
        self.model_tester = ModelTester()

        if torch.cuda.is_available():
            print("The GPU is available and will be used for computation.")
        else:
            print("The GPU is NOT available.")

        self.data_dir = utils.load_config("PATH", "DATA_DIR")
        self.tb_path = os.path.join(self.data_dir, "..", "runs")
        self.max_depth = utils.load_config("TRAINING_INFO", "MAX_DEPTH")
        self.balance = utils.load_config("DATA", "BALANCE")
        self.learning_rate = utils.load_config("TRAINING_INFO", "LEARNING_RATE")
        self.num_epochs = utils.load_config("TRAINING_INFO", "NUM_EPOCHS")
        self.test_leaves = utils.load_config("DATA", "TEST_LEAVES")
        self.validation_leaves = utils.load_config("DATA", "VALIDATION_LEAVES")
        self.train_leave_numbers = utils.leaf_training_list(
            self.test_leaves + self.validation_leaves
        )
        self.tree_channels = utils.load_config("TRAINING_INFO", "CHANNELS")

    def define_nn_functions(self):
        """Returns model, criterion, optimizer"""
        model = NeuralNet(self.number_of_channels).to(self.device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        return model, criterion, optimizer

    def loop_nobatch(self):
        """Main training loop. All data is loaded at once before the beginning of the loop."""
        date = datetime.today().strftime("%Y-%m-%d,%H:%M")
        exp_path = os.path.join(self.tb_path, "MLP", "MLP_" + date)
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
        model, criterion, optimizer = self.define_nn_functions()
        step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.97)
        print(
            f"train set shape = {X_train.shape}; validation set shape =  {X_val.shape}"
        )
        for epoch in tqdm(range(self.num_epochs), desc="training", unit="epoch"):

            # validation
            model.eval()
            with torch.no_grad():
                y_pred_val = model(X_val)
                val_loss = criterion(y_pred_val, y_val).item()
            # Train
            model.train(True)
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            training_loss = loss.item()

            loss.backward()
            optimizer.step()
            step_lr_scheduler.step()
            optimizer.zero_grad()

            self.writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": training_loss, "Validation": val_loss},
                epoch + 1,
            )
            self.writer.add_scalar(
                "learning_rate",
                step_lr_scheduler.optimizer.param_groups[0]["lr"],
                epoch,
            )
            if (epoch + 1) % (max(self.num_epochs // 15, 1)) == 0:
                print(
                    f"epoch: {epoch+1}, training_loss = {training_loss:.4f}, val_loss = {val_loss:.4f}"
                )

        model.save_nn(
            file_name=f"{date}_MLP_{self.num_epochs}epochs_lr:{self.learning_rate}_{self.number_of_channels}features_balanced:{self.balance}_.pth"
        )
        self.nn_results(model, ex_vect=X_train)

    def nn_results(self, model, ex_vect):
        """Saves model performance to tensorboard and prints it"""
        with torch.no_grad():
            # writer.add_image('mnist_images', img_grid) (to add an image
            self.writer.add_graph(
                model, ex_vect[0, :]
            )  # Don't know what it does exactly
            x_set, y_set = self.data_formatter.load_data(leaf_numbers=self.test_leaves)
            X_test, y_test = self.data_formatter.scale_and_split_data(x_set, y_set)
            # Print model performance
            y_predicted = model(X_test)
            y_test = y_test.to("cpu").numpy().flatten()
            y_predicted = y_predicted.to("cpu").numpy().flatten()
            self.writer.add_pr_curve("recall curve", y_test, y_predicted)

            metrics_dictionary = self.model_tester.performance_2class(
                y_test, y_predicted
            )
            hparam_dict = {
                "number of epochs": self.num_epochs,
                "number of features": self.number_of_channels,
                "balance dataset": self.balance,
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
        metrics_dictionary = self.model_tester.performance_2class(y_test, y_pred)
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
    trainer = Train()
    trainer.loop_nobatch()

    trainer.decision_tree()
