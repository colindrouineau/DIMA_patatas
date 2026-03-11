import spectral as sp1
import os
from matplotlib import pyplot as plt
import matplotlib
from PIL import Image


matplotlib.use("TkAgg")


class OpenImage:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def hsi_array(self, leaf):
        """Returns hyperspectral image array"""
        leaf_number, side = leaf.split("_")[0], leaf.split("_")[1]
        path = os.path.join(self.data_dir, "HSI", leaf_number, side, leaf + ".hdr")
        SpecLib = sp1.envi.open(path)
        HSIArray = SpecLib.asarray()
        return HSIArray

    def lab_array(self, leaf):
        """Returns lab label image array"""
        leaf_number, side = leaf.split("_")[0], leaf.split("_")[1]
        path = os.path.join(
            self.data_dir, "Lab_Feb2025_Mask", leaf_number, side, leaf + ".png"
        )
        lab_img = Image.open(path)
        return lab_img

    def mask_dist_array(self, leaf):
        """Returns distance to sick pixel image array"""
        leaf_number, side = leaf.split("_")[0], leaf.split("_")[1]
        path = os.path.join(
            self.data_dir, "Mask_Distance", leaf_number, side, leaf + "_dist.png"
        )
        dist_img = Image.open(path)
        return dist_img

    def show_channel(self, leaf, channel_number):
        """Shows image for chosen channel"""
        HSIArray = self.hsi_array(leaf)
        imChannel = HSIArray[:, :, channel_number]
        plt.imshow(imChannel)
        plt.title(f"Image of Channel {channel_number} of leaf {leaf}")
        plt.colorbar()
        plt.show()

    def show_pixel_spec(self, leaf, x, y):
        """Shows spectrogram for chosen pixel"""
        HSIArray = self.hsi_array(leaf)
        spectre_xy = HSIArray[x, y, :]
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
        plt.show()

    def show_dist_img(self, leaf):
        """Shows image of distance to lab label for a leaf"""
        lab_img = self.mask_dist_array(leaf)
        plt.imshow(lab_img)
        plt.title(f"Image of distance to sick pixel of leaf {leaf}")
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    data_dir = "./data"
    leaf_name = "foliolo1_enves_a4"

    channel_number = 50
    x, y = 150, 100

    open_im = OpenImage(data_dir)
    open_im.show_channel(leaf_name, channel_number)
    open_im.show_pixel_spec(leaf_name, x, y)
    open_im.show_lab_img(leaf_name)
    open_im.show_dist_img(leaf_name)
