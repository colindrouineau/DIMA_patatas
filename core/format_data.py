import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

import torch

from open_image import OpenImage
import utils


class DataFormatter:
    """
    class to format data for training
    """

    def __init__(
        self,
        device=None,
        number_of_channels: int = utils.load_config("DATA", "NUMBER_OF_CHANNELS"),
        data_type="lab_mask",
    ):
        """
        :param str data_type: labeling data type
        """
        self.open_im = OpenImage(number_of_channels=number_of_channels)
        self.number_of_channels = number_of_channels
        self.balance = utils.load_config("DATA", "BALANCE")
        self.device = device
        self.data_type = data_type

    def leaf_mask_data(self, leaf, return_mask=False):
        """Filters pixels on the leaf and format data to a list.
        Takes pixel which are on both HSI leaf and lab_img leaf.

        :param str leaf: name of the leaf
        :param bool return_mask: if True, returns (label_array, leaf_mask)


        Returns
        -------
        X_leaf_pixels : list of arrays
            The spectograms for each pixel
        Y_leaf_pixels : list
            The label for each pixel
        """
        hsi_array = self.open_im.hsi_array(leaf)
        if self.data_type == "lab_mask":
            lab_arr = self.open_im.lab_array(leaf)
            # Remove all pixel which have a too low max intensity. (< 0.01) (outside leaf)
            mask_lab = lab_arr > 0.01
        if self.data_type == "dist_mask":
            lab_arr = self.open_im.mask_dist_array(leaf)
            mask_lab = (
                lab_arr > -1
            )  # We take all, assuming hsi filter is enough, since the sick pixels have the same value than pixels outside the leaf
        
        mask_hsi = hsi_array.max(axis=-1) > 0.01
        leaf_mask = mask_hsi & mask_lab

        if return_mask:
            return lab_arr, leaf_mask
        x_leaf_pixels = hsi_array[leaf_mask]
        y_leaf_labels = lab_arr[leaf_mask]
        if self.data_type == "lab_mask":
            # label = 1 if the pixel is sick, 0 otherwise
            y_leaf_labels = np.where(y_leaf_labels == 200, 1, 0)

        return x_leaf_pixels, y_leaf_labels

    def reconstitute_leaf(self, leaf, y_pred):
        """
        Reconstitutes predicted label array to initial leaf geometry

        Returns
        -------
        y_real, y_pred : (np.array, np.array)
            images of real and predicted label
        """
        y_real, leaf_mask = self.leaf_mask_data(leaf, return_mask=True)
        # check if the labeling is continuous or 0,1

        if self.data_type == "lab_mask":
            category = list(np.unique(y_pred).astype(int)) == [0, 1]
            if category:
                # put y_pred to 0, 200, 255 format like y_real
                y_pred = np.where(y_pred == 1, 200, 255)
            else:
                # 0 = out of leaf. Fill the whole possible range of values ([0,255])
                y_pred = (y_pred + 0.05) * (255 / 1.05)

        # track mask transformation :
        height, width = leaf_mask.shape
        position_arr = np.array([[(x, y) for y in range(width)] for x in range(height)])
        # becomes a 1D arr, but we tracked position transformations
        masked_position_arr = position_arr[leaf_mask]
        # reconstitute 2 D array
        y_pred_leaf = np.zeros((height, width))
        # add missing values
        for label, (x, y) in zip(y_pred, masked_position_arr):
            y_pred_leaf[x, y] = label

        return y_real, y_pred_leaf

    def load_data(self, channels=None, leaf_numbers=None):
        """Load data.
        Set number of samples and number of features as attributes.

        :param list | None channels: if channels is not None, selects channels (accordingly to the list `channels`).
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
        verbose = len(leaves) > 50
        if verbose:
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
        if verbose:
            print(
                f"There are {n_samples} pixels in the loaded dataset with each {n_features} channels"
            )
            if self.data_type == "lab_mask":
                print(f"The proportion of bad pixels is {100 * np.mean(y_set):.2f} %")
            if self.data_type == "dist_mask":
                print(f"The mean distance to a sick pixel is {np.mean(y_set):.2f}")
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
        self,
        x_set,
        y_set,
        to_tensor=True,
        scale=True,
        requires_grad: bool = False,
    ) -> tuple:
        """Fits the data for Neural Network training. Optional parameters to specify data type and transformation."""
        # Add duplicates in the training set to have 50/50 distribution of sick/non sick pixels
        if self.balance:
            x_set, y_set = self.balance_data(x_set, y_set)
        if scale:
            sc = StandardScaler()
            x_set = sc.fit_transform(x_set)
        if to_tensor:
            x_set = torch.from_numpy(x_set.astype(np.float32)).to(self.device)
            y_set = torch.from_numpy(y_set.astype(np.float32)).to(self.device)
            y_set = y_set.view(y_set.shape[0], 1)
        if requires_grad:
            x_set.requires_grad_(True)

        return x_set, y_set


if __name__ == "__main__":
    NUMBER_OF_CHANNELS = 10
    LEAF_NAME = "foliolo2_enves_a4"

    data_format = DataFormatter(NUMBER_OF_CHANNELS)
    X, y = data_format.leaf_mask_data(LEAF_NAME)
    # taking y_real as test y_pred
    y_real, y_pred = data_format.reconstitute_leaf(LEAF_NAME, y_pred=y)

    from viz_image import VizImage

    visualise = VizImage()
    visualise.plot_y_real_pred(y_real, y_pred, title=LEAF_NAME)
