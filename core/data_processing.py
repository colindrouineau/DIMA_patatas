import numpy as np
from PIL import Image
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
import os
from format_data import DataFormatter
import utils


class ImageCleaner:
    """class to erase some of the images' irregularities"""

    def __init__(self, min_space=6, n_iter=2):
        self.save_dir = "/home/colind/work/Mines/TR_DIMA/DIMA_code/SAVE"
        self.min_space = min_space
        self.n_iter = n_iter
        self.format_data = DataFormatter()

    def cut_in_line(self, path: str, p1: tuple, p2: tuple, side: str, inplace=False):
        """Cut (put to 0) the part of the image which is on side `side` of the line crossing `p1` and `p2`"""
        image = np.array(Image.open(path))
        x1, y1 = p1
        x2, y2 = p2
        assert x2 - x1 != 0, "the cut line is vertical, and the slope is not defined."
        a = (y2 - y1) / (x2 - x1)
        assert (
            a != 0
        ), "the cut line is horizontal, therefore the side can't be interpreted"
        b = y1 - a * x1

        def line(x):
            return a * x + b

        height, width = image.shape
        erased_pixel = 0
        for x in range(height):
            for y in range(width):
                if side == "right" and a > 0 or side == "left" and a < 0:
                    if y > line(x):
                        if image[x, y] != 0:
                            erased_pixel += 1
                        image[x, y] = 0
                if side == "right" and a < 0 or side == "left" and a > 0:
                    if y < line(x):
                        if image[x, y] != 0:
                            erased_pixel += 1
                        image[x, y] = 0

        # image[x1, y1] = 150
        # image[x2, y2] = 150
        if inplace:
            save_path = path
        else:
            save_path = os.path.join(self.save_dir, path.split("/")[-1])
        image = Image.fromarray(image)
        image.save(save_path)
        print(f"Image cut successfully and saved at {save_path}")
        print(f"{erased_pixel} pixels were erased.")

    def cut_stem_image(self, path, test=False):
        """Applies `cut_stem_iter` to the image a certain number of time and saves the image."""
        image = np.array(Image.open(path))
        erased_pixel = 0

        for _ in range(self.n_iter):
            image, new_erased_pixel = self.cut_stem_iter(image)
            erased_pixel += new_erased_pixel

        if test:
            path = os.path.join(self.save_dir, path.split("/")[-1])
        Image.fromarray(image).save(path)
        print(
            f'Stem cut successful. {erased_pixel} pixels were deleted on leaf {path.split("/")[-1]}'
        )

    def cut_stem_iter(self, image: np.ndarray) -> tuple[np.ndarray, int]:
        """Erases pixels if they don't belong to a large enough connected region."""
        height, width = image.shape

        def in_leaf_length(iter_1, iter_2, horiz=True):
            """Returns array with consecutive non-zero pixel counts."""
            arr = np.zeros_like(image)
            for i in iter_1:
                count = int(0)
                for j in iter_2:
                    if horiz:
                        x, y = i, j
                    else:
                        x, y = j, i
                    if image[x, y] == 0:
                        count = 0
                    else:
                        count += 1
                        arr[x, y] = min(
                            count, 100
                        )  # because the type is uint8. It's ok because we don't need to count so high
            return arr

        # Initialize arrays for 4-directional scans
        left_in_leaf = in_leaf_length(range(height), range(width))
        up_in_leaf = in_leaf_length(range(width), range(height), horiz=False)
        right_in_leaf = in_leaf_length(range(height), list(reversed(range(width))))
        down_in_leaf = in_leaf_length(
            range(width), list(reversed(range(height))), horiz=False
        )

        erased_pixel = 0
        for x in range(height):
            for y in range(width):
                if (
                    image[x, y] > 0
                    and min(
                        left_in_leaf[x, y] + right_in_leaf[x, y] - 1,
                        up_in_leaf[x, y] + down_in_leaf[x, y] - 1,
                    )
                    <= self.min_space
                ):
                    erased_pixel += 1
                    image[x, y] = 0

        return image, erased_pixel

    # OBSOLETE
    def normalise_signal(self, X, plot=False):
        """smooths and applies SNV to the X signal"""
        smoothed = savgol_filter(X, window_length=7, polyorder=3)
        std = np.std(smoothed)
        if std == 0:
            snv = np.zeros(len(X))
        else:
            snv = (smoothed - np.mean(smoothed)) / std
        if plot:
            channels = list(range(len(X)))
            plt.plot(channels, X, label="Spectrogram before processing", color="blue")
            plt.plot(channels, smoothed, label="smoothed", color="red")
            plt.plot(channels, snv, label="smooth + snv", color="green")
            plt.legend()
            plt.show()
        return snv
    
    def normalise_image_spectra(self, leaf):
        hsi_leaf, _ = self.format_data.leaf_mask_data(leaf)
        # Apply smooth signal to all spectograms :
        normalized = np.apply_along_axis(
            self.normalise_signal,
            axis=1,
            arr=hsi_leaf,
        )
        _, hsi_arr = self.format_data.reconstitute_leaf(leaf, normalized)
        return hsi_arr

    def normalise_image(self, leaf):
        def smooth(Y):
            return savgol_filter(Y, window_length=7, polyorder=3)

        def snv(X):
            std = np.std(X)
            if std == 0:
                snv = np.zeros(len(X))
            else:
                snv = (X - np.mean(X)) / std
            return snv

        def to_0_1(X):
            return X / np.max(np.abs(X))

        # Reshape hsi_arr to 2D: (height * width, spectral_bands)
        hsi_leaf, _ = self.format_data.leaf_mask_data(leaf)
        # Apply smooth signal to all spectograms :
        smoothed = np.apply_along_axis(
            smooth,
            axis=1,
            arr=hsi_leaf,
        )
        # Apply snv on each channel
        normalized = np.apply_along_axis(
            snv,
            axis=0,
            arr=smoothed,
        )
        # Apply snv on each channel
        normalized = np.apply_along_axis(
            to_0_1,
            axis=0,
            arr=normalized,
        )
        # Reshape back to original dimensions
        _, hsi_arr = self.format_data.reconstitute_leaf(leaf, normalized)
        return hsi_arr

    def cut_all_stems(self, folder="Lab_Feb2025_Mask"):
        """Apply `cut_stem_image` to all the leaves in Lab mask dir

        Possible values for `folder` : "Lab_Feb2025_Mask", "MaskDistance"""
        path_to_folder_lab = os.path.join(utils.load_config("PATH", "DATA_DIR"), folder)
        leaves = os.listdir(path_to_folder_lab)
        for leaf in leaves:
            leaf_path = os.path.join(path_to_folder_lab, leaf, "enves")
            for image in os.listdir(leaf_path):
                self.cut_stem_image(os.path.join(leaf_path, image))

    def vector_important_features(self, X: np.ndarray[(111,), float]) -> tuple[float]:
        """return important features of a full spectrogram X

        Returns
        -------
        - min in channels 20-40
        - max in channels 40-60
        - mean in channels 60-80"""
        maxi1 = np.max(X[0:20])
        mini = np.min(X[20:40])
        maxi2 = np.max(X[40:60])
        return maxi1 - mini, maxi2 - mini

    def dataset_important(self, data):
        feature = np.apply_along_axis(
            self.vector_important_features,
            axis=1,
            arr=data,
        )
        return feature


if __name__ == "__main__":
    MIN_SPACE = 6
    N_ITER = 2
    img_cleaner = ImageCleaner(MIN_SPACE, N_ITER)
    """
    # Cuts that are already made. (for intruder leaf)
    
    PATH = "/home/colind/work/Mines/TR_DIMA/DIMA_code/data/Lab_Feb2025_Mask/foliolo3/enves/foliolo3_enves_a8.png"
    P1 = (131, 56)
    P2 = (223, 61)

    img_cleaner.cut_in_line(PATH, P1, P2, "left")

    PATH = "/home/colind/work/Mines/TR_DIMA/DIMA_code/data/Lab_Feb2025_Mask/foliolo3/enves/foliolo3_enves_a5.png"
    P1 = (161, 57)
    P2 = (220, 61)

    img_cleaner.cut_in_line(PATH, P1, P2, "left")

    P1 = (223, 73)
    P2 = (232, 96)

    PATH = "/home/colind/work/Mines/TR_DIMA/DIMA_code/data/Lab_Feb2025_Mask/foliolo5/enves"
    files = os.listdir(PATH)
    for file in files:
        path = os.path.join(PATH, file)
        img_cleaner.cut_in_line(path, P1, P2, "left", inplace=True)
"""

    # PATH = "/home/colind/work/Mines/TR_DIMA/DIMA_code/data/Lab_Feb2025_Mask_arch/foliolo2/enves/foliolo2_enves_a5.png"
    #
    # img_cleaner.cut_stem_image(PATH)

    # FOLDER = None
    # img_cleaner.cut_allstems()
    from open_image import OpenImage

    open_image = OpenImage()
    LEAF = "foliolo2_haz_a9"
    hsi_array = open_image.hsi_array(leaf=LEAF)
    img_cleaner.dataset_important(hsi_array.reshape(-1, 111))

    x, y = 150, 100
    spectrogram = hsi_array[x, y, :]
    img_cleaner.normalise_signal(spectrogram, plot=True)
