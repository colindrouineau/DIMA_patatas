import numpy as np
from open_image import OpenImage
import utils


class DataFormatter:
    """
    class to format data for training
    """

    def __init__(self, number_of_channels=utils.load_config("DATA", "NUMBER_OF_CHANNELS")):
        self.open_im = OpenImage(number_of_channels=number_of_channels)

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


if __name__ == "__main__":
    NUMBER_OF_CHANNELS = 10
    LEAF_NAME = "foliolo2_enves_a4"

    data_format = DataFormatter(NUMBER_OF_CHANNELS)
    data_format.leaf_mask_data(LEAF_NAME)
