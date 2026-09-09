import numpy as np
import matplotlib.pyplot as plt

from data_mod.format_data import DataFormatter
from data_mod.data_transformation import ProcessImage
from data_mod.viz_image import COLORS


class DataAnalyse:
    """class to compute spectras' interesting properties"""

    def __init__(self):
        self.format_data = DataFormatter()
        self.data_process = ProcessImage()

    def plot_abs_difference(self, x_0, x_1):
        mean_0 = np.mean(x_0, axis=0)
        mean_1 = np.mean(x_1, axis=0)
        difference = np.abs(mean_0 - mean_1)
        plt.plot(np.arange(0, len(difference)), difference)
        plt.title("Absolute difference between the tow classes")
        plt.show()

    def plot_spectra(self, class_list, label_list):
        """plots on the same figure the average and envelope spectra for
        - False negative
        - False positive
        - True positive
        - True negative

        But can also be used for other classes. (eg sick / healthy)
        """
        channels = self.format_data.channels
        title = ""
        total_pixels = np.sum([len(pixel_class) for pixel_class in class_list])

        for pixel_class, label, color in zip(class_list, label_list, COLORS):
            quantiles = np.percentile(pixel_class, [2.5, 97.5], axis=0)
            if len(class_list) <= 2:
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

    def open_and_plot(self):
        X, y = self.format_data.load_data()
        X0 = X[y == 0]
        X1 = X[y == 1]
        self.plot_spectra([X0, X1], ["sane", "ring"])



if __name__ == "__main__":
    data_analyst = DataAnalyse()
    data_analyst.open_and_plot()


# From last test with 1 million randomly picked pixels, important channels are :
# [ 0  1  2  3  4 65 64  5 66 63 67 62 68  6 69 61 70  7 71 27 28 60 72 26 29]
