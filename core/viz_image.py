import glob
import os
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib.animation as animation

from PIL import Image
import numpy as np
import spectral as sp1

from open_image import OpenImage
import utils


class VizImage:
    """
    Class to visualise leaf images
    """

    def __init__(
        self, number_of_channels=utils.load_config("DATA", "NUMBER_OF_CHANNELS")
    ):
        self.open_im = OpenImage(number_of_channels=number_of_channels)
        self.data_dir = utils.load_config("PATH", "DATA_DIR")
        self.colors = np.array(
            [
                "#1f77b4",  # Blue
                "#ff7f0e",  # Orange
                "#2ca02c",  # Green
                "#d62728",  # Red
                "#9467bd",  # Purple
                "#8c564b",  # Brown
                "#e377c2",  # Pink
                "#7f7f7f",  # Gray
                "#bcbd22",  # Olive
                "#17becf",  # Cyan
            ]
        )

    def show_channel(self, leaf, channel_number):
        """Shows image for chosen channel"""
        hsi_array = self.open_im.hsi_array(leaf)
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
        hsi_array = self.open_im.hsi_array(leaf)
        spectre_xy = hsi_array[x, y, :]
        plt.plot(list(range(len(spectre_xy))), spectre_xy)
        plt.xlabel("Channel")
        plt.ylabel("Intensity")
        plt.title(f"Spectogram of pixel ({x}, {y}) of leaf {leaf}")
        plt.show()

    def show_multiple_pixel_spec(self, leaf, pixels, labels):
        """Shows spectrogram for chosen pixels and add corresponding labels"""
        hsi_array = self.open_im.hsi_array(leaf)
        channels = list(range(hsi_array.shape[2]))
        for i, (pixel, label) in enumerate(zip(pixels, labels)):
            spectre_xy = hsi_array[*pixel, :]
            plt.plot(channels, spectre_xy, label=label, color=self.colors[i])
        plt.xlabel("Channel")
        plt.ylabel("Intensity")
        plt.title(f"Spectograms on leaf {leaf}")
        plt.legend()
        plt.show()

    def show_lab_img(self, leaf, red_pixel=None):
        """Shows image of lab labeling for a leaf"""
        lab_img = self.open_im.lab_array(leaf)
        if red_pixel is not None:
            x, y = red_pixel
            [
                lab_img[x, y],
                lab_img[x + 1, y],
                lab_img[x - 1, y],
                lab_img[x, y + 1],
                lab_img[x, y - 1],
            ] = [100] * 5
        plt.imshow(lab_img)
        plt.title(f"Image of lab label of leaf {leaf}")
        plt.colorbar()
        plt.show()

    def show_dist_img(self, leaf):
        """Shows image of distance to lab label for a leaf"""
        dist_img = self.open_im.mask_dist_array(leaf)
        plt.imshow(dist_img)
        plt.title(f"Image of distance to sick pixel of leaf {leaf}")
        plt.colorbar()
        plt.show()

    def show_leaf_evol(
        self,
        leaf_number: int,
        side="enves",
        red_pixel: tuple | None = None,
        channel: int | None = None,
    ):
        """Show and save animation of the temporal evolution of a given leaf

        :param tuple red_pixel: if not None, position of a pixel that will be in a different colour for all the animation
        :param None | int channel: if None, shows lab image evolution.
        Else, it must be the channel number and this function shows the HSI evolution for this channel.
        """
        folder = "Lab_Feb2025_Mask" if channel is None else "HSI"
        path_to_images = os.path.join(
            self.data_dir, folder, f"foliolo{leaf_number}", side
        )
        # Load images
        if channel is None:
            paths = utils.sort_images(glob.glob(f"{path_to_images}/*.png"))
            images = [np.array(Image.open(file)) for file in paths]
        else:
            paths = utils.sort_images(glob.glob(f"{path_to_images}/*.hdr"))
            images = [
                np.array(sp1.envi.open(file).asarray())[:, :, channel] for file in paths
            ]

        if red_pixel is not None:
            x, y = red_pixel
            for image in images:
                [
                    image[x, y],
                    image[x + 1, y],
                    image[x - 1, y],
                    image[x, y + 1],
                    image[x, y - 1],
                ] = (
                    [100] * 5 if channel is None else [1] * 5
                )
        # select time of the image from the path
        time_states = [
            path.split("/")[-1].split(".")[0].split("_")[-1] for path in paths
        ]

        # Create figure
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.axis("off")
        # Initialize with the first image
        im = ax.imshow(images[0])

        # Animation function
        def update(frame):
            im.set_array(images[frame])
            ax.set_title(
                f"Evolution of leaf {folder.split("_")[0]}_{side}_fol{leaf_number} ; time_state = {time_states[frame]} ; pix = {red_pixel}"
            )

        # Create animation
        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(images),
            interval=750,
            blit=False,
            repeat_delay=2000,
        )
        # Save or show
        os.makedirs(
            os.path.join(self.data_dir, "..", "viz", "animations"), exist_ok=True
        )
        ani.save(
            os.path.join(
                self.data_dir,
                "..",
                "viz",
                "animations",
                f"{folder.split("_")[0]}_{side}_fol{leaf_number}_ani.gif",
            ),
            writer="pillow",
            fps=1,
        )
        plt.show()

    def show_pixel_evol(self, leaf_number: int, x: int, y: int, side="enves"):
        """Shows the evolution of spectrogram for chosen pixel for chosen side"""
        path_to_images = os.path.join(
            self.data_dir, "HSI", f"foliolo{leaf_number}", side
        )
        # Load images
        leaves = os.listdir(path_to_images)
        # remove extension and duplicates
        leaves = list(set([leaf.split(".")[0] for leaf in leaves]))
        leaves = utils.sort_images(leaves)

        # select time of the image from the leaf
        time_states = [leaf.split("_")[-1] for leaf in leaves]
        hsi_arrays = [self.open_im.hsi_array(leaf) for leaf in leaves]
        spectres_xy = [hsi_array[x, y, :] for hsi_array in hsi_arrays]
        colours = np.linspace(0, 1, len(spectres_xy))
        cmap = plt.get_cmap("viridis")

        for frame, spectre in enumerate(spectres_xy):
            plt.plot(
                list(range(len(spectre))),
                spectre,
                label=time_states[frame],
                color=cmap(colours[frame]),
            )
        plt.xlabel("Channel")
        plt.ylabel("Intensity")
        plt.title(f"Spectogram of pixel ({x}, {y}) of leaf {leaf_number}_{side}")
        plt.legend()
        plt.show()

    def overlap_img(self, leaf, channel):
        """Plots the overlay of lab img of a leaf and its HSI for a chosen channel"""
        hsi_arr = self.open_im.hsi_array(leaf)
        lab_im = self.open_im.lab_array(leaf)
        plt.imshow(hsi_arr[:, :, channel])
        plt.imshow(lab_im, alpha=0.4)
        plt.title(leaf)
        plt.show()

    def plot_y_real_pred(self, y_real, y_pred, title=None):
        """shows side by side predicted and real label."""
        plt.subplot(1, 2, 1)
        plt.imshow(y_real)
        plt.title(f"real label")
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(y_pred)
        plt.title(f"predicted label")
        plt.colorbar()

        plt.suptitle(title)

        plt.show()


if __name__ == "__main__":
    LEAF_NAME = "foliolo3_enves_a6"
    NUMBER_OF_CHANNELS = -1

    im_viz = VizImage(number_of_channels=NUMBER_OF_CHANNELS)

    CHANNEL_NUMBER = 70
    x, y = 213, 108
    PIXELS = [(241, 110), (199, 123), (211, 120), (176, 92), (213, 108), (216, 86)]
    LABELS = ["stem", "sick", "ring", "sane", "main_vein", "side_vein"]

    im_viz.show_multiple_pixel_spec(LEAF_NAME, PIXELS, LABELS)

    # im_viz.show_channel(LEAF_NAME, CHANNEL_NUMBER)
    # im_viz.show_pixel_spec(LEAF_NAME, x, y)
    # im_viz.show_lab_img(LEAF_NAME, red_pixel=(x, y))
    # im_viz.show_dist_img(LEAF_NAME)
#
# # im_viz.show_leaf_evol(1, red_pixel=(x, y))
# # im_viz.show_pixel_evol(1, x, y)
# # im_viz.overlap_img(LEAF_NAME, CHANNEL_NUMBER)
# LEAF_NUMBER = 12
# im_viz.show_leaf_evol(LEAF_NUMBER, red_pixel=(x, y), channel=CHANNEL_NUMBER)
# im_viz.show_leaf_evol(LEAF_NUMBER, red_pixel=(x, y), channel=None)
