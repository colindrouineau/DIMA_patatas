import glob
import os
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider

from PIL import Image
import numpy as np
import spectral as sp1

from data_mod.open_image import OpenImage
from data_mod.data_processing import ProcessImage
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
        self,
    ):
        self.open_im = OpenImage()
        self.data_dir = utils.load_config("PATH", "DATA_DIR")
        self.data_process = ProcessImage()

    def show_channel(
        self, leaf, channel_number, normalise=False, threshold=None, noise=False
    ):
        """Shows image for chosen channel"""
        hsi_arr = self.open_im.hsi_array(leaf)
        if normalise:  # apply normalise to all pixels
            hsi_arr = self.data_process.normalise_image_spectra(hsi_arr)
        im_channel = hsi_arr[:, :, channel_number]
        if noise:
            im_channel_noise = im_channel + noise * np.random.normal(
                loc=0.0, scale=1.0, size=im_channel.shape
            )
            plt.imshow(im_channel_noise)
            plt.title(f"Image of Channel {channel_number} of leaf {leaf} with noise factor = {noise}")
            plt.colorbar()
            plt.show()
        if threshold is not None:
            # 1 = sane, 2 = sick, 0 = out of leaf
            im_channel = np.where(
                im_channel > threshold, 1, np.where(im_channel == 0, 0, 2)
            )

        plt.imshow(im_channel)
        plt.title(f"Image of Channel {channel_number} of leaf {leaf}")
        plt.colorbar()
        plt.show()

    def show_leaf_evol(
        self,
        leaf_number: int,
        side="enves",
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

        time_states = [
            path.split("/")[-1].split(".")[0].split("_")[-1] for path in paths
        ]

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        plt.subplots_adjust(bottom=0.2)  # Make space for the slider

        fig.suptitle(f"fol{leaf_number} at time: {time_states[0]}")
        # Display the first image on both subplots (or adapt as needed)
        im1 = axs[0].imshow(hsi_images[0], cmap="viridis")
        axs[0].axis("off")

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

    def spectrogram_interactive_mapping(self, channel_number, leaf, normalise=False):
        hsi_arr = self.open_im.hsi_array(leaf)
        # Select a channel to display
        channel_image = hsi_arr[:, :, channel_number]

        # Create figure and subplots
        fig, (ax_image, ax_spectrum) = plt.subplots(1, 2, figsize=(10, 5))
        plt.subplots_adjust(wspace=0.4)

        # Display the selected channel
        im = ax_image.imshow(channel_image)
        ax_image.set_title(f"Channel {channel_number} of leaf {leaf}")

        # Initialize the spectrum subplot
        (line,) = ax_spectrum.plot([], [])
        ax_spectrum.set_title("Pixel Spectrum")
        y_lim = (-1.5, 2)
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
                if normalise:
                    spectrum = self.data_process.normalise_signal(spectrum)
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
    LEAF_NAME = "foliolo2_enves_a9"

    im_viz = VizImage()

    CHANNEL_NUMBER = 73

    im_viz.show_channel(LEAF_NAME, 20, noise=0.02)

    im_viz.spectrogram_interactive_mapping(CHANNEL_NUMBER, LEAF_NAME, normalise=False)
    #
    # im_viz.show_channel(LEAF_NAME, CHANNEL_NUMBER, normalise=True, threshold=1)

    LEAF_NUMBER = 4
    im_viz.show_leaf_evol(LEAF_NUMBER, channel=CHANNEL_NUMBER)
