import os
import spectral as sp1

from matplotlib import pyplot as plt
import matplotlib
from PIL import Image
import numpy as np
import utils

matplotlib.use("TkAgg")


class OpenImage:
    """
    Class to open the leaf data and plot it
    """

    def __init__(
        self,
        data_dir=utils.load_config("PATH", "DATA_DIR"),
        number_of_channels=utils.load_config("DATA", "NUMBER_OF_CHANNELS"),
    ):
        """
        :param str data_dir: path of the data directory
        :param int | str number_of_channels: For computation speed, it may be useful not to take all channel.
            Default to "all". There are 111 channels. If higher value is given, `number_of_channels` will be set to 111
        """
        self.data_dir = data_dir
        self.number_of_channels = number_of_channels

    def hsi_array(self, leaf):
        """Returns hyperspectral image array

        :param str leaf: name of the leaf
        """
        leaf_number, side = leaf.split("_")[0], leaf.split("_")[1]
        path = os.path.join(self.data_dir, "HSI", leaf_number, side, leaf + ".hdr")
        spec_lib = sp1.envi.open(path)
        hsi_arr = spec_lib.asarray()
        if self.number_of_channels in list(range(1, 112)):
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

    def leaves(self, enves_only=True):
        """Returns a sorted list of all the leaf names in the db,
        containing haz only if not `enves_only`."""
        leaf_names = []
        hsi_path = os.path.join(self.data_dir, "HSI")
        leaves = os.listdir(hsi_path)
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
        return sorted(leaf_names)

    def show_channel(self, leaf, channel_number):
        """Shows image for chosen channel"""
        hsi_array = self.hsi_array(leaf)
        if channel_number >= hsi_array.shape[2]:
            print(
                f"You try to print channel n°{channel_number} when there are only {hsi_array.shape[2]} channels. Showing last channel instead."
            )
            channel_number = hsi_array.shape[2] - 1
        im_channel = hsi_array[:, :, channel_number]
        plt.imshow(im_channel)
        plt.title(f"Image of Channel {channel_number} of leaf {leaf}")
        plt.colorbar()
        plt.show()

    def show_pixel_spec(self, leaf, x, y):
        """Shows spectrogram for chosen pixel"""
        hsi_array = self.hsi_array(leaf)
        spectre_xy = hsi_array[x, y, :]
        plt.plot(list(range(len(spectre_xy))), spectre_xy)
        plt.xlabel("Channel")
        plt.ylabel("Intensity")
        plt.title(f"Spectogram of pixel ({x}, {y}) of leaf {leaf}")
        plt.show()

    def show_lab_img(self, leaf):
        """Shows image of lab labeling for a leaf"""
        lab_img = self.lab_array(leaf)
        plt.imshow(lab_img)
        plt.title(f"Image of lab label of leaf {leaf}")
        plt.colorbar()
        plt.show()

    def show_dist_img(self, leaf):
        """Shows image of distance to lab label for a leaf"""
        dist_img = self.mask_dist_array(leaf)
        plt.imshow(dist_img)
        plt.title(f"Image of distance to sick pixel of leaf {leaf}")
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    LEAF_NAME = "foliolo2_enves_a4"
    NUMBER_OF_CHANNELS = -1

    open_im = OpenImage(number_of_channels=NUMBER_OF_CHANNELS)

    CHANNEL_NUMBER = 80
    x, y = 150, 100

    hsi_ex = open_im.hsi_array(LEAF_NAME)
    print(f"HSI image has dimensions : {hsi_ex.shape}")

    open_im.show_channel(LEAF_NAME, CHANNEL_NUMBER)
    open_im.show_pixel_spec(LEAF_NAME, x, y)
    open_im.show_lab_img(LEAF_NAME)
    open_im.show_dist_img(LEAF_NAME)

    print(open_im.leaves())
