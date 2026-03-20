import numpy as np
import matplotlib.pyplot as plt
from format_data import DataFormatter
import utils


class DataAnalyse:
    """class to compute images' interesting properties"""

    def __init__(self):
        self.format_data = DataFormatter(number_of_channels=utils.load_config("DATA", "TOTAL_N_CHANNELS"))

    def find_significant_channels(self):
        """Channels where the difference between sick pixel and healthy pixel is the higher"""
        # load all data
        x_set, y_set = self.format_data.load_data()
        print(x_set.shape, y_set.shape)
        sick_pixels = x_set[y_set == 1]
        healthy_pixels = x_set[y_set == 0]

        sick_mean = np.mean(sick_pixels, axis=0)
        healthy_mean = np.mean(healthy_pixels, axis=0)

        difference = (sick_mean - healthy_mean) ** 2
        channel_order = np.argsort(difference)
        NUMBER_OF_IMPORTANT_CHANNELS = 5
        important_channels = channel_order[-NUMBER_OF_IMPORTANT_CHANNELS:]
        important_difference = difference[important_channels]

        channels = list(range(self.format_data.number_of_channels))

        quantiles_healthy = np.percentile(healthy_pixels, [2.5, 97.5], axis=0)
        plt.fill_between(
                channels,
                quantiles_healthy[0],
                quantiles_healthy[1],
                color="green",
                alpha=0.15,
                label=f"Healthy pixel 95% confidence envelope",
            )

        quantiles_sick = np.percentile(sick_pixels, [2.5, 97.5], axis=0)
        plt.fill_between(
                channels,
                quantiles_sick[0],
                quantiles_sick[1],
                color="orange",
                alpha=0.15,
                label=f"Sick pixel 95% confidence envelope",
            )
    
        print(important_channels)
        print(important_difference)
        plt.plot(channels, healthy_mean, label='healthy mean', color="green")
        plt.plot(channels, sick_mean, label="sick mean", color="orange")
        plt.title(f"Channel intensity distribution for sick and healthy pixel. N_pixels = {len(x_set)}, {len(sick_pixels) / len(x_set) * 100:.2f} % sick, {len(healthy_pixels) / len(x_set) * 100:.2f} % healthy")
        plt.xlabel("channel")
        plt.ylabel("intensity")
        plt.legend()
        plt.show()

        return important_channels, important_difference


if __name__ == '__main__':
    data_analyst = DataAnalyse()

    print(data_analyst.find_significant_channels())