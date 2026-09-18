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

    def hsi_array(self, leaf, channels=None):
        """Returns hyperspectral image array

        :param str leaf: name of the leaf
        :param list channels:
        """
        leaf_number, side = leaf.split("_")[0], leaf.split("_")[1]
        path = os.path.join(self.data_dir, "HSI", leaf_number, side, leaf + ".hdr")
        spec_lib = sp1.envi.open(path)
        hsi_arr = spec_lib.asarray()
        if channels is not None:
            hsi_arr = hsi_arr[:, :, channels]
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
            self.data_dir, "Mask_RelDist", leaf_number, side, leaf + ".png"
        )
        dist_img = Image.open(path)
        return np.array(dist_img)

    def temp_array(self, leaf):
        """Returns temporal distance to sick pixel image array"""
        leaf_number, side = leaf.split("_")[0], leaf.split("_")[1]
        path = os.path.join(
            self.data_dir, "Temporal_Mask", leaf_number, side, leaf + ".png"
        )
        dist_img = Image.open(path)
        return np.array(dist_img)

    def leaves(self, enves_only=True, leaf_numbers=None):
        """Returns a sorted list of all the leaf names in the db,
        containing haz only if not `enves_only`. Does not return the last images of each leaf because it doesn't exist for Temporal Mask

        :param list | None leaf_number: if is None, returns all leaves, else the ones in the list
        """
        leaf_names = []
        folder_path = os.path.join(self.data_dir, "Temporal_Mask")
        leaves = (
            os.listdir(folder_path)
            if leaf_numbers is None
            else [f"foliolo{leaf_number}" for leaf_number in leaf_numbers]
        )
        for leaf in leaves:
            time_series = os.listdir(os.path.join(folder_path, leaf, "enves"))
            # remove extension and duplicates
            time_series = list(
                set([time_leaf.split(".")[0] for time_leaf in time_series])
            )
            leaf_names += time_series
            if not enves_only:
                time_series = os.listdir(os.path.join(folder_path, leaf, "haz"))
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

