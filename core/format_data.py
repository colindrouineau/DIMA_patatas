import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch

from open_image import OpenImage
import utils


class DataFormatter:
    """
    class to format data for training
    """

    def __init__(
        self, device, number_of_channels=utils.load_config("DATA", "NUMBER_OF_CHANNELS")
    ):
        self.open_im = OpenImage(number_of_channels=number_of_channels)
        self.test_leaves = utils.load_config("DATA", "TEST_LEAVES")
        self.number_of_leaves = utils.load_config("DATA", "NUMBER_OF_LEAVES")
        self.train_leave_numbers = utils.leaf_training_list(self.test_leaves)
        self.number_of_channels = number_of_channels
        self.balance = utils.load_config("DATA", "BALANCE")
        self.device = device

    def leaf_mask_data(self, leaf):
        """Filters pixels on the leaf and format data to a list.
        Takes pixel which are on both HSI leaf and lab_img leaf.

        :param str leaf: name of the leaf

        Returns
        -------
        X_leaf_pixels : list of arrays
            The spectograms for each pixel
        Y_leaf_pixels : list
            The label for each pixel
        """
        hsi_array = self.open_im.hsi_array(leaf)
        lab_arr = self.open_im.lab_array(leaf)
        # Remove all pixel which have a too low max intensity. (< 0.01)
        mask_hsi = hsi_array.max(axis=-1) > 0.01
        mask_lab = lab_arr > 0.01
        leaf_mask = mask_hsi & mask_lab
        x_leaf_pixels = hsi_array[leaf_mask]
        y_leaf_labels = lab_arr[leaf_mask]
        # label = 1 if the pixel is sick, 0 otherwise
        y_leaf_labels = np.where(y_leaf_labels == 200, 1, 0)
        return x_leaf_pixels, y_leaf_labels

    def load_data(self, channels=None, leaf_numbers=None):
        """Load data.
        Set number of samples and number of features as attributes.

        :param list | None channel: if channel is not None, selects channels (accordingly to the list `channel`).
        If it is None, selects channels as indicated in CONFIG file.
        :param list | None leaf_number: if None, all leaves data should be loaded. If list, only the corresponding leaves.

        Returns
        -------
        x_set : np.array
            Pixels array. Dim (number_of_samples, number_of_channels = features)
        y_set : np.array
            Labels array. Dim (number_of_samples)
        """
        leaves = self.open_im.leaves(leaf_numbers=leaf_numbers)
        all_training_leaves = (
            False
            if leaf_numbers is None
            else len(leaf_numbers) == self.number_of_leaves - len(self.test_leaves)
        )
        if all_training_leaves:
            leaves = tqdm(leaves, desc="loading data", unit="leaf")
        if channels is None:
            x_set = np.empty((0, self.number_of_channels))
        else:
            x_set = np.empty((0, len(channels)))
        y_set = []

        for leaf in leaves:
            x, y = self.leaf_mask_data(leaf)
            if channels is not None:
                x = x[:, channels]
            x_set = np.concat((x_set, x))
            y_set = np.concat((y_set, y))

        n_samples, n_features = x_set.shape
        if all_training_leaves:
            print(
                f"There are {n_samples} pixels in the loaded dataset with each {n_features} channels"
            )
            print(f"The proportion of bad pixels is {100 * np.mean(y_set):.2f} %")
        return x_set, y_set

    def balance_data(self, x, y):
        """Add randomly duplicates in the (training) set to have 50/50 distribution of sick/non sick pixels.
        Then shuffle."""
        pos_n = int(np.sum(y))
        n = y.shape[0]
        # How many sick pixels we should add to have a proportion of self.balance of sick pixel in the train dataset
        gap = int((self.balance * n - pos_n) / (1 - self.balance))
        if gap <= 0:
            print(
                f"There are already enough sick pixels for a balanced data set (balance={self.balance})"
            )
            return x, y

        positive_profiles = x[y == 1]
        added_positive_idx = np.random.choice(range(pos_n), size=gap, replace=True)
        added_positive = np.copy(positive_profiles[added_positive_idx])

        x = np.concat((x, added_positive))
        y = np.concat((y, np.ones(gap)))
        # Finally, shuffle
        suffle_idx = np.random.choice(range(x.shape[0]), size=x.shape[0], replace=False)

        return x[suffle_idx], y[suffle_idx]

    def scale_and_split_data(
        self, x_set, y_set, to_tensor=True, scale=True, trainer: bool | None = None
    ) -> tuple:
        """Fits the data for Neural Network training.

        :param None | bool trainer:
        - if None, splits data using train_test_split and returns train and test data.
        - if True, use all x_set and y_set as train data
        - if False, use all x_set and y_set as test data

        Returns
        -------
        X_train, X_test, y_train, y_test"""
        if trainer is None:
            X_train, X_test, y_train, y_test = train_test_split(
                x_set,
                y_set,
                test_size=utils.load_config("DATA", "TEST_SIZE"),
                random_state=1,
                stratify=y_set,
            )
        if trainer is True:
            X_train, y_train = x_set, y_set  # should I shuffle ?
        if trainer is False:
            X_test, y_test = x_set, y_set

        # Add duplicates in the training set to have 50/50 distribution of sick/non sick pixels
        if self.balance and (trainer is None or trainer is True):
            X_train, y_train = self.balance_data(X_train, y_train)
        if scale:
            sc = StandardScaler()
            if trainer is not False:
                X_train = sc.fit_transform(X_train)
            if trainer is not True:
                X_test = sc.fit_transform(X_test)

        if to_tensor:
            if trainer is not False:
                X_train = torch.from_numpy(X_train.astype(np.float32)).to(self.device)
                X_train.requires_grad_(True)
                y_train = torch.from_numpy(y_train.astype(np.float32)).to(self.device)
                y_train = y_train.view(y_train.shape[0], 1)
            if trainer is not True:
                X_test = torch.from_numpy(X_test.astype(np.float32)).to(self.device)
                y_test = torch.from_numpy(y_test.astype(np.float32)).to(self.device)
                y_test = y_test.view(y_test.shape[0], 1)

        if trainer is True:
            return X_train, y_train
        if trainer is False:
            return X_test, y_test
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    NUMBER_OF_CHANNELS = 10
    LEAF_NAME = "foliolo2_enves_a4"

    data_format = DataFormatter(NUMBER_OF_CHANNELS)
    data_format.leaf_mask_data(LEAF_NAME)
