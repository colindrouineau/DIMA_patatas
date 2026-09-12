import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle

import torch

from data_mod.open_image import OpenImage
from data_mod.data_transformation import ProcessImage
from data_mod.operation_on_spectra import SpectraOperation
import utils


class DataFormatter:
    """
    class to format data for training
    """

    def __init__(self, test=False, balance_data=True):
        """
        initiates attributes using CONFIG information
        """
        self.open_im = OpenImage()
        self.device = torch.device(utils.load_config("TRAINING_INFO", "DEVICE"))
        self.data_type = utils.load_config("TRAINING_CHOICE", "DATA_TYPE")
        self.model_type = utils.load_config("TRAINING_CHOICE", "MODEL_TYPE")
        self.channels = utils.load_config("TRAINING_CHOICE", "CHANNELS")
        if self.channels == "all":
            self.channels = list(range(111))
        self.image_process = ProcessImage()
        self.test = test
        self.balance_data = balance_data

    def leaf_mask_data(self, leaf, return_mask=False):
        """Filters pixels on the leaf and format data to a list.
        Takes pixel which are on both HSI leaf and lab_img leaf.
        Selects and labels pixels accordingly to the data type

        :param str leaf: name of the leaf
        :param bool return_mask: if True, returns (label_array, leaf_mask)
        :param bool temporal: if True, uses temporal label for selection


        Returns
        -------
        X_leaf_pixels : list of arrays
            The spectograms for each pixel
        Y_leaf_pixels : list
            The actual label for each pixel
        """
        hsi_array = self.open_im.hsi_array(leaf, channels=self.channels)
        lab_arr = self.open_im.lab_array(leaf)

        dist_arr = self.open_im.mask_dist_array(leaf)
        temp_arr = self.open_im.temp_array(leaf)

        # pixels which are too sick :
        too_sick = dist_arr > 128 + 5
        # pixels which are in the ring :
        in_the_ring_dist = dist_arr < 10

        # pixels which are too sick :
        # too_sick = temp_arr < 128 - 3
        # pixels which are in the ring :
        in_the_ring_temp = temp_arr == 128
        in_the_ring = in_the_ring_dist | in_the_ring_temp

        # pixels which are in the leaf :
        in_the_leaf = lab_arr > 0.01  # for labels
        mask_hsi = hsi_array.max(axis=-1) > 0.01  # for data
        is_sick = lab_arr == 200

        if self.data_type == "lab_mask":
            if self.test:
                mask = in_the_leaf
            else:
                # Select pixels in the leaf, not too sick, and not in the ring
                mask = (in_the_leaf & ~too_sick) & ~in_the_ring
            # label = 1 if the pixel is sick, 0 otherwise
            label_arr = np.where(lab_arr == 200, 1, 0)
        if self.data_type in ["dist_mask", "temp_mask"]:
            # Select pixels in the leaf, not sick
            mask = in_the_leaf & ~is_sick
            label_arr = lab_arr
            label_arr[in_the_ring] = 1
            label_arr[~in_the_ring] = 0

        leaf_mask = mask_hsi & mask
        if return_mask:
            return lab_arr, leaf_mask
        x_leaf_pixels = hsi_array[leaf_mask]
        y_leaf_labels = label_arr[leaf_mask]

        return x_leaf_pixels, y_leaf_labels

    def reconstitute_leaf(self, leaf, arr):
        """
        Reconstitutes one array of dimension 1 or 2 that was filtered through leaf_mask to initial leaf geometry.
        It can be either labels or hsi_array.

        Returns
        -------
        y_real, to_leaf_form : (np.array, np.array)
            images of real label (y_real is zeroed outside the mask) and reconstituted leaf
        """
        y_real, mask = self.leaf_mask_data(leaf, return_mask=True)

        # track mask transformation :
        height, width = mask.shape
        position_arr = np.array([[(x, y) for y in range(width)] for x in range(height)])
        # becomes a 1D arr, but we tracked position transformations
        masked_position_arr = position_arr[mask]

        dimension = len(arr.shape)
        # for lab_mask label
        # check if the labeling is continuous or 0,1
        if dimension == 1:
            to_leaf_form = np.zeros((height, width))
        elif dimension == 2:
            bands = arr.shape[1]
            to_leaf_form = np.zeros((height, width, bands))

        # reconstitute 2 or 3 D array
        # add missing values
        for element, (x, y) in zip(arr, masked_position_arr):
            if dimension == 1:
                to_leaf_form[x, y] = element
            if dimension == 2:
                to_leaf_form[x, y, :] = element

        y_real[~mask] = 0

        return y_real, to_leaf_form

    def load_data(self, leaf_numbers=None):
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
        if (
            leaf_numbers is not None and type(leaf_numbers[0]) == str
        ):  # then a particular leaf is selected
            leaves = leaf_numbers
        else:
            leaves = self.open_im.leaves(leaf_numbers=leaf_numbers)
        verbose = len(leaves) > 50
        if verbose:
            leaves = tqdm(leaves, desc="loading data", unit="leaf")

        x_set = np.empty((0, len(self.channels)))
        y_set = []

        for leaf in leaves:
            x, y = self.leaf_mask_data(leaf)
            x_set = np.concat((x_set, x))
            y_set = np.concat((y_set, y))

        n_samples, n_features = x_set.shape

        # shuffle data
        x_set, y_set = shuffle(x_set, y_set)
        if self.balance_data:
            y0 = y_set[y_set == 0]
            X0 = x_set[y_set == 0]
            y1 = y_set[y_set == 1]
            X1 = x_set[y_set == 1]
            n0 = len(y0)
            n1 = len(y1)
            np.random.seed(1)
            if n0 > n1:
                rd_0elements = np.random.choice(y0.shape[0], size=n1, replace=False)
                selected_y0 = y0[rd_0elements]
                selected_X0 = X0[rd_0elements, :]
                y_set = np.concatenate((selected_y0, y1))
                x_set = np.concatenate((selected_X0, X1))
            if n1 > n0:
                rd_1elements = np.random.choice(y1.shape[0], size=n0, replace=False)
                selected_y1 = y1[rd_1elements,]
                selected_X1 = y1[rd_1elements, :]
                y_set = np.concatenate((selected_y1, y0))
                x_set = np.concatenate((selected_X1, X0))
        # shuffle data
        x_set, y_set = shuffle(x_set, y_set)
        if verbose:
            print(
                f"There are {n_samples} pixels in the loaded dataset with each {n_features} channels"
            )
            print(
                f"The proportion of bad (sick or soon sick) pixels is {100 * np.mean(y_set):.2f} %"
            )
        return x_set, y_set

    def scale_and_format_data(
        self,
        x_set,
        y_set,
        to_tensor=True,
        scale=False,
        requires_grad: bool = False,
    ) -> tuple:
        """Fits the data for Neural Network training. Optional parameters to specify data type and transformation."""
        # Add duplicates in the training set to have 50/50 distribution of sick/non sick pixels
        if scale:
            x_set = SpectraOperation().normalise(x_set)
        if to_tensor:
            x_set = torch.from_numpy(x_set.astype(np.float32)).to(self.device)
            y_set = torch.from_numpy(y_set.astype(np.float32)).to(self.device)
            dim2 = 1 if len(y_set.shape) == 1 else y_set.shape[1]
            y_set = y_set.view(y_set.shape[0], dim2)
        if requires_grad:
            x_set.requires_grad_(True)

        return x_set, y_set


if __name__ == "__main__":
    LEAF_NAME = "foliolo2_enves_a4"

    data_format = DataFormatter()
    x_set, y_set = data_format.load_data()
    x_set, y_set = data_format.scale_and_format_data(
        x_set, y_set, to_tensor=False, scale=True
    )

    X, y = data_format.leaf_mask_data(LEAF_NAME)
    # taking y_real as test y_pred
    y_real, y_pred = data_format.reconstitute_leaf(LEAF_NAME, arr=y)

    from data_mod.viz_image import VizImage

    visualise = VizImage()
    visualise.plot_y_real_pred(y_real, y_pred, title="test only real : " + LEAF_NAME)
