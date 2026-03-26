import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from format_data import DataFormatter
import utils
from open_image import OpenImage
from viz_image import COLORS


class DataAnalyse:
    """class to compute images' interesting properties"""

    def __init__(
        self, number_of_channels=utils.load_config("DATA", "TOTAL_N_CHANNELS")
    ):
        self.format_data = DataFormatter(number_of_channels=number_of_channels)

    def find_significant_channels(self):
        """Channels where the difference between sick pixel and healthy pixel is the higher"""
        # load all data
        x_set, y_set = self.format_data.load_data()
        print(x_set.shape, y_set.shape)
        sick_pixels = x_set[y_set == 1]
        healthy_pixels = x_set[y_set == 0]

        sick_mean = np.mean(sick_pixels, axis=0)
        healthy_mean = np.mean(healthy_pixels, axis=0)

        difference = np.abs(sick_mean - healthy_mean)
        channel_order = np.argsort(difference)
        NUMBER_OF_IMPORTANT_CHANNELS = 5
        important_channels = channel_order[-NUMBER_OF_IMPORTANT_CHANNELS:]
        important_difference = difference[important_channels]

        self.plot_spectra([sick_pixels, healthy_pixels], ['sick_pixels', 'healthy_pixels'])

        print("The important channels are : ", important_channels)
        print("The corresponding |healthy_mean - sick_mean| : ", important_difference)
        return important_channels, important_difference

    def plot_spectra(self, class_list, label_list):

        channels = list(range(self.format_data.number_of_channels))
        title = ""
        total_pixels = np.sum([len(pixel_class) for pixel_class in class_list])

        for pixel_class, label, color in zip(class_list, label_list, COLORS):

            quantiles = np.percentile(pixel_class, [2.5, 97.5], axis=0)
            plt.fill_between(
                channels,
                quantiles[0],
                quantiles[1],
                color=color,
                alpha=0.15,
                label=f"{label} 95% confidence envelope",
            )

            class_mean = np.mean(pixel_class, axis=0)

            plt.plot(channels, class_mean, label=f"{label} mean", color=color)
            title += f"{label} = {100 * len(pixel_class) / total_pixels:.2f} %,  "

        plt.title("Channel intensity distribution. " + title)
        plt.xlabel("channel")
        plt.ylabel("intensity")
        plt.legend()
        plt.show()

    def alternative_distance_mask(self, leaf):
        """
        Compute the distance from each point to the closest point of the other class (0 or 1), ignoring -1.

        Args:
            arr: 2D array of -1, 0, and 1.

        Returns:
            distance_map: 2D array where each element is the distance to the closest point of the other class.
                          -1 values remain unchanged.
        """
        lab_mask = OpenImage().lab_array(leaf)
        arr = np.where(lab_mask == 0, -1, np.where(lab_mask == 200, 1, 0))
        # Create masks for 0s and 1s, ignoring -1
        mask_0 = (arr == 0)
        mask_1 = (arr == 1)

        # Distance from 1s to nearest 0s (ignoring -1)
        distance_to_0 = np.zeros_like(arr, dtype=float)
        if np.any(mask_0):
            distance_to_0 = distance_transform_edt(~mask_0)

        # Distance from 0s to nearest 1s (ignoring -1)
        distance_to_1 = np.zeros_like(arr, dtype=float)
        if np.any(mask_1):
            distance_to_1 = distance_transform_edt(~mask_1)

        # Combine: for each point, take the distance to the other class
        distance_map = np.zeros_like(arr, dtype=float)
        distance_map[mask_0] = distance_to_1[mask_0]
        distance_map[mask_1] = - distance_to_0[mask_1]
        distance_map[arr == -1] = - 127  # if out of the leaf

        return distance_map

if __name__ == "__main__":
    data_analyst = DataAnalyse()

    # data_analyst.find_significant_channels()

    from open_image import OpenImage
    LEAF = "foliolo10_enves_a12"
    imager = OpenImage()
    lab_mask = imager.lab_array(LEAF)
    lab_mask = np.where(lab_mask == 0, -1, np.where(lab_mask == 200, 1, 0))
    dist_mask = data_analyst.alternative_distance_mask(lab_mask)
    plt.imshow(dist_mask)
    plt.colorbar()
    plt.show()