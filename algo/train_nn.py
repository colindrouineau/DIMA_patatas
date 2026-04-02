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
from torch.utils.data import DataLoader, TensorDataset
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
        self.model_tester = ModelTester(
            model_path=None
        )  # We don't need to load a model, just performance method

        if torch.cuda.is_available():
            print("The GPU is available and will be used for computation.")
        else:
            print("The GPU is NOT available.")

        self.data_dir = utils.load_config("PATH", "DATA_DIR")
        self.exp_name = self.date + "_" + self.model_type
        self.tb_path = os.path.join(
            self.data_dir, "..", "runs", self.data_type, self.exp_name
        )

        os.makedirs(self.tb_path, exist_ok=True)
        self.writer = SummaryWriter(self.tb_path)
        self.balance = utils.load_config("TRAINING_CHOICE", "BALANCE")
        self.validation_leaves = utils.load_config("DATA", "VALIDATION_LEAVES")
        self.train_leave_numbers = utils.leaf_training_list()
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
        X_train, y_train = self.data_formatter.scale_and_format_data(
            X_train, y_train, to_device=False
        )
        X_val, y_val = self.data_formatter.load_data(
            leaf_numbers=self.validation_leaves
        )
        X_val, y_val = self.data_formatter.scale_and_format_data(X_val, y_val, to_device=False)

        self.define_nn_functions()

        print(
            f"train set shape = {X_train.shape}; validation set shape =  {X_val.shape}"
        )
        BATCH_SIZE = utils.load_config("DATA", "BATCH_SIZE")
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=12, persistent_workers=True, pin_memory=True)
        val_dataset = TensorDataset(X_val, y_val)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=12, persistent_workers=True, pin_memory=True)

        return train_loader, val_loader

    def epoch_info(self, epoch, training_loss, val_loss):
        """Logs and prints useful information for epoch"""
        self.writer.add_scalar("Training loss", training_loss, epoch + 1)
        self.writer.add_scalar("Validation loss", val_loss, epoch + 1)
        self.writer.add_scalar(
            "Learning_rate", self.step_lr_scheduler.get_last_lr()[0], epoch + 1
        )
        if (epoch + 1) % (max(self.num_epochs // 30, 1)) == 0 or epoch == 0:
            print(
                f"epoch: {epoch+1}, training_loss = {training_loss:.4f}, val_loss = {val_loss:.4f}, lr = {self.step_lr_scheduler.get_last_lr()[0]:.4f}"
            )

    def one_epoch(self, train_loader, val_loader):
        # Each epoch has a training and validation phase
        training_loss, val_loss = 0, 0
        for phase in ["train", "val"]:
            if phase == "train":
                self.model.train()  # Set model to training mode
                dataloader = train_loader
            else:
                self.model.eval()  # Set model to evaluate mode
                dataloader = val_loader

            running_loss = 0.0

            for X, y_real in dataloader:
                X = X.to(self.device)
                y_real = y_real.to(self.device)
                with torch.set_grad_enabled(phase == "train"):
                    y_pred = self.model(X)
                    loss = self.criterion(y_pred, y_real)

                    if phase == "train":
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                # statistics
                running_loss += loss.item() * X.size(0)
            
            epoch_loss = running_loss / len(dataloader)

            if phase == "train":
                training_loss = epoch_loss
                self.step_lr_scheduler.step(training_loss)
            else:
                val_loss = epoch_loss

        return training_loss, val_loss

    def main_loop(self):
        """Main training loop. All data is loaded at once before the beginning of the loop."""
        train_loader, val_loader = self.loop_initialiser()

        for epoch in tqdm(range(self.num_epochs), desc="training", unit="epoch"):
            training_loss, val_loss = self.one_epoch(train_loader, val_loader)
            self.epoch_info(epoch, training_loss, val_loss)
            # Check early stopping
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print(
                    f"Early stopping triggered at epoch {epoch}. Last (val_loss, train_loss) = {round(val_loss, 4), round(training_loss, 4)}"
                )
                break

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
        self.model.save_nn(
            self.early_stopping.best_model_state,
            nn_trace=self.nn_trace,
            file_name=f"{self.exp_name}.pth",
        )

    def nn_results(self):
        """Saves model performance to tensorboard and prints it"""
        with torch.no_grad():
            # writer.add_image('mnist_images', img_grid) (to add an image

            x_set, y_set = self.data_formatter.load_data(
                leaf_numbers=self.validation_leaves
            )
            X_val, y_val = self.data_formatter.scale_and_format_data(x_set, y_set)
            self.writer.add_graph(self.model, X_val)
            self.nn_trace = jit.trace(self.model, X_val)

            # Print model performance
            y_predicted = self.model(X_val)
            y_val = y_val.to("cpu").numpy()
            y_predicted = y_predicted.to("cpu").numpy()
            print(f"Performance of model {self.exp_name} on validation dataset")
            metrics_dictionary, y_predicted, y_val = self.model_tester.performance(
                y_val, y_predicted
            )
            save = input("Do you want to save this model ? (Y/n)")
            if save not in ["", "y", "Y"]:
                self.writer.close()
                log_dir = os.path.join(
                    self.data_dir, "..", "runs", self.data_type, self.exp_name
                )
                confirm = input(
                    f"You are about to delete the folder '{log_dir}' and all the files and folders it contains. Are you sure ? (type 'rm' to delete)"
                )
                if confirm == "rm":
                    shutil.rmtree(log_dir)
                sys.exit()
            # PR curve makes sense only for 2 class classification problems
            if self.data_type == "lab_mask":
                self.writer.add_pr_curve("recall curve", y_val, y_predicted)

            hparam_dict = {
                "number of epochs": self.num_epochs,
                "number of features": self.number_of_channels,
                "balance dataset": self.balance,
                "initial lr": self.learning_rate,
                "normalised": utils.load_config("TRAINING_CHOICE", "NORMALISE"),
            }
            self.writer.add_hparams(
                hparam_dict=hparam_dict,
                metric_dict=metrics_dictionary,
                run_name=self.tb_path,
            )
            training_info = utils.load_config(
                "TRAINING_INFO", self.data_type.upper(), self.model_type.upper()
            )
            training_info = str(training_info)
            self.writer.add_text(
                tag="model additional tuning", text_string=training_info
            )
            training_functions = f"Model is : {self.model}, \n Loss function is : {self.criterion}, \n Optimizer is {self.optimizer}"
            self.writer.add_text(tag="model functions", text_string=training_functions)
            self.writer.close()


if __name__ == "__main__":
    # To choose training type, change CONFIG file.
    trainer = TrainNN()
    trainer.main_loop()
