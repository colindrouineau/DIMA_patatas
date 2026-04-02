import numpy as np
import matplotlib.pyplot as plt

from data_mod.format_data import DataFormatter
from data_mod.data_processing import ProcessImage
import utils
from data_mod.open_image import OpenImage
from data_mod.viz_image import COLORS


class DataAnalyse:
    """class to compute images' interesting properties"""

    def __init__(self):
        self.format_data = DataFormatter()
        self.data_process = ProcessImage()

    def find_significant_channels(self, normalise=True):
        """Channels where the difference between sick pixel and healthy pixel is the higher"""
        # load all data except test
        x_set, y_set = self.format_data.load_data(leaf_numbers=utils.leaf_no_test())
        # normalise data
        ### TO BE TAKEN OFF
        x_set = x_set[:100000, :]
        y_set = y_set[:100000]
        ### TO BE TAKEN OFF
        if normalise:
            x_set = self.data_process.normalise_image_spectra(x_set)

        sick_pixels = x_set[y_set == 1]
        healthy_pixels = x_set[y_set == 0]

        sick_mean = np.mean(sick_pixels, axis=0)
        healthy_mean = np.mean(healthy_pixels, axis=0)

        difference = np.abs(sick_mean - healthy_mean)
        channel_order = np.argsort(difference)
        NUMBER_OF_IMPORTANT_CHANNELS = 25
        important_channels = channel_order[-NUMBER_OF_IMPORTANT_CHANNELS:]
        important_difference = difference[important_channels]
        important_difference = [float(round(idif, 4)) for idif in important_difference]
        print("The important channels are : ", important_channels[::-1])
        print("The corresponding |healthy_mean - sick_mean| : ", important_difference[::-1])

        self.plot_spectra(
            [sick_pixels, healthy_pixels], ["sick_pixels", "healthy_pixels"]
        )
        self.plot_abs_difference(difference, ["sick_pixels", "healthy_pixels"])

        return important_channels[::-1], important_difference[::-1]

    def plot_spectra(self, class_list, label_list):
        """plots on the same figure the average and envelope spectra for
        - False negative
        - False positive
        - True positive
        - True negative
        """
        channels = list(range(self.format_data.number_of_channels))
        title = ""
        total_pixels = np.sum([len(pixel_class) for pixel_class in class_list])

        for pixel_class, label, color in zip(class_list, label_list, COLORS):

            quantiles = np.percentile(pixel_class, [2.5, 97.5], axis=0)
            plt.plot(
                channels,
                quantiles[0],
                color=color,
                label=f"{label} 95% confidence envelope",
                linestyle="--",
            )
            plt.plot(channels, quantiles[1], color=color, linestyle="--")
            plt.fill_between(
                channels, quantiles[0], quantiles[1], color=color, alpha=0.1
            )

            class_mean = np.mean(pixel_class, axis=0)
            plt.plot(channels, class_mean, label=f"{label} mean", color=color)
            title += f"{label} = {100 * len(pixel_class) / total_pixels:.2f} %,  "

        plt.title("Channel intensity distribution. " + title)
        plt.xlabel("channel")
        plt.ylabel("intensity")
        plt.legend()
        plt.show()

    def plot_abs_difference(self, difference, label_list):
        channels = list(range(self.format_data.number_of_channels))
        title = f"Absolute difference between {label_list[0]} and {label_list[1]}"
        plt.plot(channels, difference)
        plt.title(title)
        plt.show()

    def vector_important_features(self, X: np.ndarray[(111,), float]) -> tuple[float]:
        """return important features of a full spectrogram X

        Returns
        -------
        - min in channels 20-40
        - max in channels 40-60
        - mean in channels 60-80"""
        assert len(X) > 40, f"Array X has a length = {len(X)}, which is not enough to extract features."
        maxi1 = np.max(X[0:20])
        mini = np.min(X[20:40])
        maxi2 = np.max(X[40:60])
        return maxi1 - mini, maxi2 - mini

    def dataset_important(self, data: np.ndarray) -> np.ndarray:
        """Return important features for all the vectors in the data"""
        feature = np.apply_along_axis(
            self.vector_important_features,
            axis=1,
            arr=data,
        )
        return feature


if __name__ == "__main__":
    data_analyst = DataAnalyse()
    data_analyst.find_significant_channels()

# From last test with 1 million randomly picked pixels, important channels are :
# [ 0  1  2  3  4 65 64  5 66 63 67 62 68  6 69 61 70  7 71 27 28 60 72 26 29]