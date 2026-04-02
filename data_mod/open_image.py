import os
import spectral as sp1
from PIL import Image
import numpy as np

import utils


class OpenImage:
    """
    Class to open the leaf data
    """

    def __init__(self):
        self.data_dir = utils.load_config("PATH", "DATA_DIR")
        self.number_of_channels = utils.load_config("DATA", "NUMBER_OF_CHANNELS")

    def hsi_array(self, leaf):
        """Returns hyperspectral image array

        :param str leaf: name of the leaf
        """
        leaf_number, side = leaf.split("_")[0], leaf.split("_")[1]
        path = os.path.join(self.data_dir, "HSI", leaf_number, side, leaf + ".hdr")
        spec_lib = sp1.envi.open(path)
        hsi_arr = spec_lib.asarray()
        n_tot_channels = utils.load_config("DATA", "TOTAL_N_CHANNELS")
        if self.number_of_channels in list(range(1, n_tot_channels)):
            # Slice to select only some channels
            step = hsi_arr.shape[2] // self.number_of_channels
            start = (hsi_arr.shape[2] % self.number_of_channels) // 2
            end = -(hsi_arr.shape[2] % self.number_of_channels) // 2
            hsi_arr = hsi_arr[:, :, start:end:step]
        return hsi_arr

    def lab_array(self, leaf):
        """Returns lab label image array"""
        leaf_number, side = leaf.split("_")[0], leaf.split("_")[1]
        path = os.path.join(
            self.data_dir, "Lab_Feb2025_Mask", leaf_number, side, leaf + ".png"
        )
        lab_img = Image.open(path)
        return np.array(lab_img)

    def mask_dist_array(self, leaf):
        """Returns distance to sick pixel image array"""
        leaf_number, side = leaf.split("_")[0], leaf.split("_")[1]
        path = os.path.join(
            self.data_dir, "Mask_Distance", leaf_number, side, leaf + "_dist.png"
        )
        dist_img = Image.open(path)
        return np.array(dist_img)
    
    def ring_mask_array(self, leaf):
        """Returns ring mask array"""
        leaf_number, side = leaf.split("_")[0], leaf.split("_")[1]
        path = os.path.join(
            self.data_dir, "Ring_Mask_Class", leaf_number, side, leaf + ".png"
        )
        ring_img = Image.open(path)
        return np.array(ring_img)

    def leaves(self, enves_only=True, leaf_numbers=None):
        """Returns a sorted list of all the leaf names in the db,
        containing haz only if not `enves_only`.

        :param list | None leaf_number: if is None, returns all leaves, else the ones in the list
        """
        leaf_names = []
        hsi_path = os.path.join(self.data_dir, "HSI")
        leaves = (
            os.listdir(hsi_path)
            if leaf_numbers is None
            else [f"foliolo{leaf_number}" for leaf_number in leaf_numbers]
        )
        for leaf in leaves:
            time_series = os.listdir(os.path.join(hsi_path, leaf, "enves"))
            # remove extension and duplicates
            time_series = list(
                set([time_leaf.split(".")[0] for time_leaf in time_series])
            )
            leaf_names += time_series
            if not enves_only:
                time_series = os.listdir(os.path.join(hsi_path, leaf, "haz"))
                time_series = list(
                    set([time_leaf.split(".")[0] for time_leaf in time_series])
                )
                leaf_names += time_series
        return utils.sort_images(leaf_names)


if __name__ == "__main__":
    LEAF_NAME = "foliolo2_enves_a9"

    open_im = OpenImage()

    CHANNEL_NUMBER = 80
    x, y = 150, 100
    hsi_ex = open_im.hsi_array(LEAF_NAME)
    print(f"HSI image has dimensions : {hsi_ex.shape}")

    ring_ex = open_im.mask_dist_array(LEAF_NAME)
    print(ring_ex)
    print(np.unique(ring_ex), ring_ex.shape)
