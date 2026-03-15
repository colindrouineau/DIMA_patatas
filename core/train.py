import numpy as np
import os
from tqdm import tqdm

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

        if torch.cuda.is_available():
            print("The GPU is available and will be used for computation.")
        else:
            print("The GPU is NOT available.")

        self.data_dir = utils.load_config("PATH", "DATA_DIR")
        tb_path = os.path.join(self.data_dir, "..", "runs")
        os.makedirs(tb_path, exist_ok=True)
        self.writer = SummaryWriter(tb_path)
        self.max_depth = utils.load_config("TRAINING_INFO", "MAX_DEPTH")
        self.balance = utils.load_config("DATA", "BALANCE")
        self.learning_rate = utils.load_config("TRAINING_INFO", "LEARNING_RATE")
        self.num_epochs = utils.load_config("TRAINING_INFO", "NUM_EPOCHS")
        self.test_leaves = utils.load_config("DATA", "TEST_LEAVES")
        self.train_leave_numbers = utils.leaf_training_list(self.test_leaves)
        self.tree_channels = utils.load_config("TRAINING_INFO", "CHANNELS")

    def define_nn_functions(self):
        """Returns model, criterion, optimizer"""
        model = NeuralNet(self.number_of_channels).to(self.device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        return model, criterion, optimizer

    def loop_nobatch(self):
        """Main training loop. All data is loaded at once before the beginning of the loop."""
        x_set, y_set = self.data_formatter.load_data(
            leaf_numbers=self.train_leave_numbers
        )
        X_train, y_train = self.data_formatter.scale_and_split_data(
            x_set, y_set, trainer=True
        )
        model, criterion, optimizer = self.define_nn_functions()
        step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.97)

        for epoch in tqdm(range(self.num_epochs), desc="training", unit="epoch"):
            # Forward pass and loss
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            # Backward pass and update
            loss.backward()
            optimizer.step()
            step_lr_scheduler.step()
            # zero grad before new step
            optimizer.zero_grad()

            if (epoch + 1) % (max(self.num_epochs // 15, 1)) == 0:
                print(f"epoch: {epoch+1}, loss = {loss.item():.4f}")
                self.writer.add_scalar("training loss", loss.item(), epoch)

        model.save_nn(
            file_name=f"MLP_{self.num_epochs}epochs_lr:{self.learning_rate}_{self.number_of_channels}features_balanced:{self.balance}.pth"
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
            X_test, y_test = self.data_formatter.scale_and_split_data(
                x_set, y_set, trainer=False
            )
            # Print model performance
            y_predicted = model(X_test)
            y_test = y_test.to("cpu").numpy().flatten()
            y_predicted = y_predicted.to("cpu").numpy().flatten()
            self.writer.add_pr_curve("recall curve", y_test, y_predicted, global_step=0)
            self.writer.close()
            self.performance_2class(y_test, y_predicted)

    def decision_tree(self):
        """decision tree training"""
        patch_sklearn()
        x_set, y_set = self.data_formatter.load_data(
            channels=self.tree_channels, leaf_numbers=self.train_leave_numbers
        )
        X_train, y_train = self.data_formatter.scale_and_split_data(
            x_set, y_set, trainer=True, to_tensor=False, scale=False
        )

        # Create Decision Tree classifer object
        clf = DecisionTree(max_depth=self.max_depth, channels=self.tree_channels)
        # Train Decision Tree Classifer
        clf = clf.fit(X_train, y_train)
        # Predict the response for test dataset

        x_set, y_set = self.data_formatter.load_data(
            channels=self.tree_channels, leaf_numbers=self.test_leaves
        )
        X_test, y_test = self.data_formatter.scale_and_split_data(
            x_set, y_set, trainer=False, to_tensor=False, scale=False
        )
        y_pred = clf.predict(X_test)
        self.performance_2class(y_test, y_pred)
        clf.viz_decision_tree()

    def performance_2class(self, y_test, y_predicted):
        """Print performance information of a 2-class classification model"""
        y_test = y_test.astype(bool)
        y_predicted = y_predicted.round().astype(bool)
        n = y_test.shape[0]
        # type fit for mask
        y_predicted = y_predicted.astype(bool)
        y_test = y_test.astype(bool)
        true_positive = np.sum(y_predicted & y_test).astype(float)
        false_positive = np.sum(y_predicted & ~y_test).astype(float)
        true_negative = np.sum(~y_predicted & ~y_test).astype(float)
        false_negative = np.sum(~y_predicted & y_test).astype(float)
        print(
            f"Proportion of sick pixel: {100 * (true_positive + false_negative) / n:.2f} %"
        )
        print(
            f"Proportion of pixel detected as sick: {100 * (true_positive + false_positive) / n:.2f} %"
        )
        print(f"accuracy: {100 * (true_positive + true_negative) / n:.2f} %")
        print(f"TPR: {100 * true_positive / (true_positive + false_negative):.2f} %")
        print(f"FPR: {100 * false_positive / (false_positive + true_negative):.2f} %")


if __name__ == "__main__":
    trainer = Train()
    trainer.loop_nobatch()

    # trainer.decision_tree(channels=CHANNELS)
