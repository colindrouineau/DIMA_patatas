import glob
import os
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider

from PIL import Image
import numpy as np
import spectral as sp1

from open_image import OpenImage
from data_processing import ImageCleaner
import utils

COLORS = np.array(
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


class VizImage:
    """
    Class to visualise leaf images
    """

    def __init__(
        self, number_of_channels=utils.load_config("DATA", "NUMBER_OF_CHANNELS")
    ):
        self.open_im = OpenImage(number_of_channels=number_of_channels)
        self.data_dir = utils.load_config("PATH", "DATA_DIR")
        self.data_process = ImageCleaner()

    def show_channel(self, leaf, channel_number, normalise=False, threshold=None):
        """Shows image for chosen channel"""
        hsi_arr = self.open_im.hsi_array(leaf)
        if channel_number >= hsi_arr.shape[2]:
            print(
                f"You try to print channel n°{channel_number} when there are only {hsi_arr.shape[2]} channels. Showing last channel instead."
            )
            channel_number = hsi_arr.shape[2] - 1
        if normalise:  # apply normalise to all pixels
            hsi_arr = self.data_process.normalise_image_spectra(leaf)
        if threshold is None:
            im_channel = hsi_arr[:, :, channel_number]
        else:
            im_channel = np.where(np.min(hsi_arr, axis=2) < threshold, 2, 1)
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

    def show_multiple_pixel_spec(self, leaf, pixels, labels, normalise=False):
        """Shows spectrogram for chosen pixels and add corresponding labels"""
        hsi_arr = self.open_im.hsi_array(leaf)
        if normalise:  # apply normalise to all pixels
            hsi_arr = self.data_process.normalise_image(leaf, hsi_arr)
        channels = list(range(hsi_arr.shape[2]))
        for i, (pixel, label) in enumerate(zip(pixels, labels)):
            spectre_xy = hsi_arr[*pixel, :]
            plt.plot(channels, spectre_xy, label=label, color=COLORS[i])
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
        channel: int = 70,
    ):
        """Show and save animation of the temporal evolution of a given leaf

        :param tuple red_pixel: if not None, position of a pixel that will be in a different colour for all the animation
        :param int channel: shows lab image evolution.
        Else, it must be the channel number and this function shows the HSI evolution for this channel.
        """
        hsi_path = os.path.join(self.data_dir, "HSI", f"foliolo{leaf_number}", side)
        paths = utils.sort_images(glob.glob(f"{hsi_path}/*.hdr"))
        hsi_images = [
            np.array(sp1.envi.open(file).asarray())[:, :, channel] for file in paths
        ]

        lab_mask_path = os.path.join(
            self.data_dir, "Lab_Feb2025_Mask", f"foliolo{leaf_number}", side
        )
        paths = utils.sort_images(glob.glob(f"{lab_mask_path}/*.png"))
        lab_images = [np.array(Image.open(file)) for file in paths]

        if red_pixel is not None:
            x, y = red_pixel
            for image in hsi_images:
                [
                    image[x, y],
                    image[x + 1, y],
                    image[x - 1, y],
                    image[x, y + 1],
                    image[x, y - 1],
                ] = [1] * 5
        # select time of the image from the path
        time_states = [
            path.split("/")[-1].split(".")[0].split("_")[-1] for path in paths
        ]

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        plt.subplots_adjust(bottom=0.2)  # Make space for the slider

        fig.suptitle(f"fol{leaf_number} at time: {time_states[0]}")
        # Display the first image on both subplots (or adapt as needed)
        im1 = axs[0].imshow(hsi_images[0], cmap="viridis")
        axs[0].axis("off")
        axs[0].set_title(f"HSI. Pix: {red_pixel}")

        im2 = axs[1].imshow(lab_images[0], cmap="viridis")
        axs[1].axis("off")
        axs[1].set_title("Lab mask")

        # Create a slider axis
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        slider = Slider(
            ax_slider, "Frame", 0, len(lab_images) - 1, valinit=0, valstep=1
        )

        current_frame = 0

        # Update function for the slider
        def update(val):
            nonlocal current_frame
            current_frame = int(slider.val)
            fig.suptitle(f"fol{leaf_number} at time: {time_states[current_frame]}")
            im1.set_array(hsi_images[current_frame])
            im2.set_array(lab_images[current_frame])
            fig.canvas.draw_idle()

        # Function to handle key presses
        def on_key(event):
            nonlocal current_frame
            if event.key == "right":
                current_frame = min(current_frame + 1, len(lab_images) - 1)
            elif event.key == "left":
                current_frame = max(current_frame - 1, 0)
            else:
                return  # Ignore other keys

            # Update slider and redraw
            slider.set_val(current_frame)
            update(current_frame)

        # Connect the key press event
        fig.canvas.mpl_connect("key_press_event", on_key)

        # Connect the slider to the update function
        slider.on_changed(update)
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

    def plot_class_spectra(self, y_real, y_pred, title=None):
        """plots on the same figure the average and envelope spectra for
        - False negative
        - False positive
        - True positive
        - True negative
        """

    def spectrogram_interactive_mapping(self, leaf, normalise=False):
        hsi_arr = self.open_im.hsi_array(leaf)
        if normalise:  # apply normalise to all pixels
            hsi_arr = self.data_process.normalise_image(leaf, hsi_arr)
        # Select a channel to display
        selected_channel = 80  # Example: first channel
        channel_image = hsi_arr[:, :, selected_channel]

        # Create figure and subplots
        fig, (ax_image, ax_spectrum) = plt.subplots(1, 2, figsize=(10, 5))
        plt.subplots_adjust(wspace=0.4)

        # Display the selected channel
        im = ax_image.imshow(channel_image)
        ax_image.set_title(f"Channel {selected_channel} of leaf {leaf}")

        # Initialize the spectrum subplot
        (line,) = ax_spectrum.plot([], [])
        ax_spectrum.set_title("Pixel Spectrum")
        y_lim = (-1, 1) if normalise else (0, 0.7)
        ax_spectrum.set_ylim(y_lim)
        ax_spectrum.set_xlabel("channel")
        ax_spectrum.set_ylabel("intensity")

        # Store all spectra and their corresponding lines
        spectra_lines = []
        spectra_data = []
        crosses = []

        # Function to handle mouse clicks
        def on_click(event):
            nonlocal spectra_lines, spectra_data, crosses
            if event.inaxes != ax_image:
                return  # Ignore clicks outside the image subplot

                # Right click: Reset the spectrum subplot
            if event.button == 3:  # Right mouse button
                for line in spectra_lines:
                    line.remove()
                for cross in crosses:
                    for part in cross:
                        part.remove()
                spectra_lines = []
                spectra_data = []
                crosses = []
                ax_spectrum.set_title("Pixel Spectrum")
                fig.canvas.draw()
                return

            # Get the clicked pixel coordinates (rounded to nearest integer)
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)

            # Ensure the click is within the image bounds
            if 0 <= x < hsi_arr.shape[1] and 0 <= y < hsi_arr.shape[0]:
                spectrum = hsi_arr[y, x, :]
                spectra_data.append((x, y, spectrum))
                color = COLORS[len(spectra_lines) % 10]
                # Plot the new spectrum
                (line,) = ax_spectrum.plot(
                    np.arange(len(spectrum)),
                    spectrum,
                    color=color,
                    label=f"Pixel ({x}, {y})",
                )
                spectra_lines.append(line)

                # Draw a cross on the image at (x, y) with the same color
                cross_horizontal = ax_image.plot(
                    [x - 3, x + 3], [y, y], color=color, linewidth=1.5
                )
                cross_vertical = ax_image.plot(
                    [x, x], [y - 3, y + 3], color=color, linewidth=1.5
                )
                crosses.append([cross_horizontal[0], cross_vertical[0]])

                ax_spectrum.legend()
                ax_spectrum.relim()
                ax_spectrum.autoscale_view()
                ax_spectrum.set_title(f"Pixel Spectra (Last: {x}, {y})")
                fig.canvas.draw()

        # Connect the click event
        fig.canvas.mpl_connect("button_press_event", on_click)

        plt.show()


if __name__ == "__main__":
    LEAF_NAME = "foliolo7_enves_a10"
    NUMBER_OF_CHANNELS = -1

    im_viz = VizImage(number_of_channels=NUMBER_OF_CHANNELS)

    CHANNEL_NUMBER = 73
    x, y = 213, 108
    PIXELS = [(241, 110), (199, 123), (211, 120), (176, 92), (213, 108), (216, 86)]
    LABELS = ["stem", "sick", "ring", "sane", "main_vein", "side_vein"]

    # im_viz.spectrogram_interactive_mapping(LEAF_NAME, normalise=True)

    # im_viz.show_multiple_pixel_spec(LEAF_NAME, PIXELS, LABELS, normalise=True)
    im_viz.show_channel(LEAF_NAME, CHANNEL_NUMBER, normalise=True, threshold=None)
    im_viz.show_channel(LEAF_NAME, CHANNEL_NUMBER, normalise=True, threshold=0.5)
    # im_viz.show_pixel_spec(LEAF_NAME, x, y)
    # im_viz.show_lab_img(LEAF_NAME, red_pixel=(x, y))
    # im_viz.show_dist_img(LEAF_NAME)
    #
    # # im_viz.show_leaf_evol(1, red_pixel=(x, y))
    # # im_viz.show_pixel_evol(1, x, y)
    # # im_viz.overlap_img(LEAF_NAME, CHANNEL_NUMBER)
    LEAF_NUMBER = 12
    # im_viz.show_leaf_evol(LEAF_NUMBER, red_pixel=(x, y), channel=CHANNEL_NUMBER)
