import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from open_image import OpenImage
from format_data import DataFormatter
from model import NeuralNet
from tqdm import tqdm
import utils


class Train:
    """
    Main class for data loading and model training
    """

    def __init__(
        self, number_of_channels=utils.load_config("DATA", "NUMBER_OF_CHANNELS")
    ):
        """Instanciates OpenImage and DataFormatter, and set device to GPU, as attributes"""
        self.open_im = OpenImage(number_of_channels=number_of_channels)
        self.data_formatter = DataFormatter(number_of_channels=number_of_channels)
        self.number_of_channels = number_of_channels
        if torch.cuda.is_available():
            print("The GPU is available and will be used for computation.")
        else:
            print("The GPU is NOT available.")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self):
        """Load data.
        Set number of samples and number of features as attributes.

        Returns
        -------
        x_set : np.array
            Pixels array. Dim (number_of_samples, number_of_channels = features)
        y_set : np.array
            Labels array. Dim (number_of_samples)
        """
        leaves = self.open_im.leaves()
        x_set = np.empty((0, self.number_of_channels))
        y_set = []
        for leaf in tqdm(leaves, desc="loading data", unit="leaf"):
            x, y = self.data_formatter.leaf_mask_data(leaf)
            x_set = np.concat((x_set, x))
            y_set = np.concat((y_set, y))

        n_samples, n_features = x_set.shape
        print(
            f"There are {n_samples} pixels in the loaded dataset with each {n_features} channels"
        )
        print(f"The proportion of bad pixels is {100 * np.mean(y_set):.2f}")
        self.n_samples = n_samples
        self.n_features = n_features
        return x_set, y_set

    def balance_data(self, x, y):
        """Add randomly duplicates in the (training) set to have 50/50 distribution of sick/non sick pixels.
        Then shuffle."""
        pos_n = int(np.sum(y))
        neg_n = y.shape[0] - pos_n
        gap = neg_n - pos_n

        positive_profiles = x[y == 1]
        added_positive_idx = np.random.choice(range(pos_n), size=gap, replace=True)
        added_positive = np.copy(positive_profiles[added_positive_idx])

        x = np.concat((x, added_positive))
        y = np.concat((y, np.ones(gap)))
        # Finallly, shuffle
        suffle_idx = np.random.choice(range(x.shape[0]), size=x.shape[0], replace=False)

        return x[suffle_idx], y[suffle_idx]

    def scale_and_split_data(self, x_set, y_set, balance=utils.load_config("DATA", "BALANCE"), test_size=utils.load_config("DATA", "TEST_SIZE")):
        """Fits the data for Neural Network training.

        :param bool balance: if True, calls `balance_data` function for the training dataset

        Returns
        -------
        X_train, X_test, y_train, y_test"""
        X_train, X_test, y_train, y_test = train_test_split(
            x_set, y_set, test_size=test_size, random_state=1234, stratify=y_set
        )
        # Add duplicates in the training set to have 50/50 distribution of sick/non sick pixels
        if balance:
            X_train, y_train = self.balance_data(x_set, y_set)
        # scale
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        X_train = torch.from_numpy(X_train.astype(np.float32)).to(self.device)
        X_train.requires_grad_(True)
        X_test = torch.from_numpy(X_test.astype(np.float32)).to(self.device)
        y_train = torch.from_numpy(y_train.astype(np.float32)).to(self.device)
        y_test = torch.from_numpy(y_test.astype(np.float32)).to(self.device)

        y_train = y_train.view(y_train.shape[0], 1)
        y_test = y_test.view(y_test.shape[0], 1)

        return X_train, X_test, y_train, y_test

    def define_model(self):
        """Returns model, criterion, optimizer"""
        model = NeuralNet(self.n_features).to(self.device)
        learning_rate = utils.load_config("TRAINING_INFO", "LEARNING_RATE")
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        return model, criterion, optimizer

    def loop(self, num_epochs=utils.load_config("TRAINING_INFO", "NUM_EPOCHS")):
        """Main training loop"""
        x_set, y_set = self.load_data()
        X_train, X_test, y_train, y_test = self.scale_and_split_data(x_set, y_set)
        model, criterion, optimizer = self.define_model()
        for epoch in range(num_epochs):
            # Forward pass and loss
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)

            # Backward pass and update
            loss.backward()
            optimizer.step()

            # zero grad before new step
            optimizer.zero_grad()

            if (epoch + 1) % (max(num_epochs // 15, 1)) == 0:
                print(f"epoch: {epoch+1}, loss = {loss.item():.4f}")

        with torch.no_grad():
            # Print model performance
            y_predicted = model(X_test)
            y_test = y_test.to("cpu").numpy().flatten().astype(bool)
            y_predicted = y_predicted.round().to("cpu").numpy().flatten().astype(bool)
            n = y_test.shape[0]

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
            print(
                f"TPR: {100 * true_positive / (true_positive + false_negative):.2f} %"
            )
            print(
                f"FPR: {100 * false_positive / (false_positive + true_negative):.2f} %"
            )


if __name__ == "__main__":
    trainer = Train()
    trainer.loop()
