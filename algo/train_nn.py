import numpy as np
import os
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from data_mod.open_image import OpenImage
from data_mod.format_data import DataFormatter
from algo.nn_models import BinPixNN, DistPixNN, RingPixNN
from algo.test_model import ModelTester
import utils
import algo.train_utils as train_utils


class TrainNN:
    """
    Main class for data loading and model training
    """

    def __init__(self):
        """
        - instanciates OpenImage and DataFormatter
        - set device to GPU, as attributes
        - set all useful info from CONFIG as attribute

        :param str model_type: by default, "MLP". Other possibilities are "CNN"

        """
        self.date = datetime.today().strftime("%d-%m--%H:%M")
        self.number_of_channels = utils.load_config("DATA", "NUMBER_OF_CHANNELS")
        self.model_type = utils.load_config("TRAINING_CHOICE", "MODEL_TYPE")
        self.data_type = utils.load_config("TRAINING_CHOICE", "DATA_TYPE")

        self.open_im = OpenImage()
        self.data_formatter = DataFormatter()
        self.model_tester = ModelTester()

        if torch.cuda.is_available():
            print("The GPU is available and will be used for computation.")
        else:
            print("The GPU is NOT available.")

        self.data_dir = utils.load_config("PATH", "DATA_DIR")
        self.tb_path = os.path.join(self.data_dir, "..", "runs", self.data_type, self.date + "_" + self.model_type)
        os.makedirs(self.tb_path, exist_ok=True)
        self.writer = SummaryWriter(self.tb_path)
        self.balance = utils.load_config("TRAINING_CHOICE", "BALANCE")
        self.test_leaves = utils.load_config("DATA", "TEST_LEAVES")
        self.validation_leaves = utils.load_config("DATA", "VALIDATION_LEAVES")
        self.train_leave_numbers = utils.leaf_training_list(
            self.test_leaves + self.validation_leaves
        )
        self.device = torch.device(utils.load_config("TRAINING_INFO", "DEVICE"))
        training_info = utils.load_config(
            "TRAINING_INFO", self.data_type.upper(), self.model_type.upper()
        )
        self.learning_rate = training_info["LEARNING_RATE"]
        self.num_epochs = training_info["NUM_EPOCHS"]
        self.delta = training_info["DELTA"]

    def define_mlp_bin_functions(self):
        training_info = utils.load_config("TRAINING_INFO", "LAB_MASK", "MLP")
        self.model = BinPixNN().to(self.device)
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
        self.early_stopping = train_utils.EarlyStopping(patience=10, delta=self.delta)

    def define_mlp_dist_functions(self):
        training_info = utils.load_config("TRAINING_INFO", "DIST_MASK", "MLP")
        self.model = DistPixNN().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.step_lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            "min",
            factor=training_info["FACTOR"],
            patience=training_info["PATIENCE"],
            threshold=training_info["THRESHOLD"],
        )
        self.early_stopping = train_utils.EarlyStopping(patience=100, delta=self.delta)

    def define_mlp_ring_functions(self):
        training_info = utils.load_config("TRAINING_INFO", "RING_MASK", "MLP")
        self.model = RingPixNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.step_lr_scheduler = ReduceLROnPlateau(
            self.optimizer,
            "min",
            factor=training_info["FACTOR"],
            patience=training_info["PATIENCE"],
            threshold=training_info["THRESHOLD"],
        )
        self.early_stopping = train_utils.EarlyStopping(patience=100, delta=self.delta)

    def define_nn_functions(self):
        """Sets model, criterion, optimizer, lr_scheduler as attributes"""
        if self.model_type == "MLP" and self.data_type == "lab_mask":
            self.define_mlp_bin_functions()
        if self.model_type == "MLP" and self.data_type == "dist_mask":
            self.define_mlp_dist_functions()
        if self.model_type == "MLP" and self.data_type == "ring_mask":
            self.define_mlp_ring_functions()

        # step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.97)

    def loop_initialiser(self):
        """Returns all useful variables to start training loop, and define training functions

        Returns
        -------
        X_train, y_train, X_val, y_val, date
        """
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
        return X_train, y_train, X_val, y_val

    def epoch_info(self, epoch, training_loss, val_loss):
        """Logs and prints useful information for epoch"""
        self.writer.add_scalar("Training", training_loss, epoch + 1)
        self.writer.add_scalar("Validation", val_loss, epoch + 1)
        self.writer.add_scalar(
            "Learning_rate", self.step_lr_scheduler.get_last_lr()[0], epoch + 1
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
        X_train, y_train, X_val, y_val = self.loop_initialiser()

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

            x_set, y_set = self.data_formatter.load_data(leaf_numbers=self.test_leaves)
            X_test, y_test = self.data_formatter.scale_and_split_data(x_set, y_set)
            self.writer.add_graph(model, X_test)
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
                "normalised": utils.load_config("TRAINING_CHOICE", "NORMALISE"),
            }
            self.writer.add_hparams(
                hparam_dict=hparam_dict, metric_dict=metrics_dictionary, run_name=self.tb_path
            )
            training_info = utils.load_config("TRAINING_INFO", self.data_type.upper(), self.model_type.upper())
            training_info = str(training_info)
            self.writer.add_text(tag="model additional tuning", text_string=training_info)
            training_functions = f"Model is : {self.model}, \n Loss function is : {self.criterion}, \n Optimizer is {self.optimizer}"
            self.writer.add_text(tag="model functions", text_string=training_functions)
            self.writer.close()


if __name__ == "__main__":
    # To choose training type, change CONFIG file.
    trainer = TrainNN()
    trainer.loop_nobatch()
