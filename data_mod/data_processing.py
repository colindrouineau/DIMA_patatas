import numpy as np
from PIL import Image
from scipy.signal import savgol_filter
from matplotlib import pyplot as plt
import os
import utils
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt


class ProcessImage:
    """class to erase some of the images' irregularities and process it."""

    def __init__(self):
        self.save_dir = "/home/colind/work/Mines/TR_DIMA/DIMA_code/SAVE"

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

        for _ in range(2):  # n_iter
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
                    <= 6  # min_space
                ):
                    erased_pixel += 1
                    image[x, y] = 0

        return image, erased_pixel

    def cut_all_stems(self, folder="Lab_Feb2025_Mask"):
        """Apply `cut_stem_image` to all the leaves in Lab mask dir

        Possible values for `folder` : "Lab_Feb2025_Mask", "MaskDistance"""
        path_to_folder_lab = os.path.join(utils.load_config("PATH", "DATA_DIR"), folder)
        leaves = os.listdir(path_to_folder_lab)
        for leaf in leaves:
            leaf_path = os.path.join(path_to_folder_lab, leaf, "enves")
            for image in os.listdir(leaf_path):
                self.cut_stem_image(os.path.join(leaf_path, image))

    def normalise_signal(self, X: np.ndarray) -> np.ndarray:
        """smooths and applies SNV to the X signal
        
        UPDATE : savgol filter is no longer used because of its computation time and because it is not deemed necessary"""
        #smoothed = savgol_filter(X, window_length=7, polyorder=3)
        smoothed = X
        std = np.std(smoothed)
        if std == 0:
            snv = np.zeros(len(X))
        else:
            snv = (smoothed - np.mean(smoothed)) / std
        return snv

    def normalise_image_spectra(self, hsi_arr: np.ndarray) -> np.ndarray:
        """Applies `normalise_signal` to every spectrogram of the array.
        Array can be a sequence or an image of spectrograms (2D or 3D)"""
        n_pixels = hsi_arr.shape[0]
        dimension = len(hsi_arr.shape)
        normalized = np.empty_like(hsi_arr)
        chunk_size = (
            n_pixels // 100
        )  # to have an advancement bar without slowing down too much operations
        for i in tqdm(
            range(0, n_pixels, chunk_size),
            desc=f"Normalising {n_pixels} " + ("pixels" if dimension == 2 else "lines"),
            unit="chunk",
        ):
            end = min(i + chunk_size, n_pixels)
            normalized[i:end] = np.apply_along_axis(
                func1d=self.normalise_signal, axis=dimension - 1, arr=hsi_arr[i:end]
            )
        return normalized

    def relative_distance_mask(self, lab_mask):
        """
        Compute the distance from each point to the closest point of the other class (0 or 1), ignoring -1. T

        Args:
            arr: 2D array of 0, 255, 200 (output of `OpenImage().lab_array`).

        Returns:
            distance_map: 2D array where each element is the distance to the closest point of the other class.
                          -1 values remain unchanged. 
                          WARNING : dtype = float
        """
        arr = np.where(lab_mask == 0, -1, np.where(lab_mask == 200, 1, 0))
        # Create masks for 0s and 1s, ignoring -1
        mask_0 = arr == 0
        mask_1 = arr == 1

        # Distance from 1s to nearest 0s (ignoring -1)
        distance_to_0 = np.zeros_like(arr, dtype=float)
        distance_to_0 = distance_transform_edt(~mask_0)

        # Distance from 0s to nearest 1s (ignoring -1)
        distance_to_1 = np.zeros_like(arr, dtype=float)
        if np.any(mask_1):
            distance_to_1 = distance_transform_edt(~mask_1)

        # Combine: for each point, take the distance to the other class
        distance_map = np.zeros_like(arr, dtype=float)
        distance_map[mask_0] = distance_to_1[mask_0]
        distance_map[mask_1] = -distance_to_0[mask_1]
        distance_map[arr == -1] = -127  # if out of the leaf

        return distance_map

    def create_class_ring_array(self, lab_mask: np.ndarray) -> np.ndarray:
        """
        :param np.ndarray lab_mask: Array filled with 0 (outside leaf), 200 (sick pixel), 255 (healthy pixel)

        Returns
        -------
        ring_mask  np.ndarray
            A new mask filled with 0 (outside leaf), 200 (sick pixel), 255 (healthy pixel), 100 (ring pixel)
        """
        # All the points that are at a distance of less than 30 pixels from a sick zone
        # Select sick points, and compute the distance to them
        mask_sick = lab_mask == 200
        distance_to_1 = np.zeros_like(lab_mask, dtype=float)
        distance_to_1 = distance_transform_edt(~mask_sick)
        # color to 100 the points that are close enough to 1 (and in the leaf)
        lab_mask[(lab_mask > 0) & (distance_to_1 <= 30) & (distance_to_1 > 0)] = 100

        return lab_mask
    
    def create_cont_ring_array(self, lab_mask):
        """
        :param np.ndarray lab_mask: Array filled with 0 (outside leaf), 200 (sick pixel), 255 (healthy pixel)

        Returns
        -------
        relative_distance : np.ndarray
            A new mask filled with 0 (outside leaf or sick pixels), 31 * 8 (healthy pixel), 0 to 30 * 8 (ring pixel, 8 * distance to sick)
        """
        relative_distance = self.relative_distance_mask(lab_mask)
        class_ring_array = self.create_class_ring_array(lab_mask)
        relative_distance[class_ring_array == 255] = 31 * 8
        relative_distance[class_ring_array == 100] *= 8
        relative_distance[(class_ring_array == 200) | (class_ring_array == 0)] = 0  # sick pixels considered as outside leaf
        relative_distance = np.round(relative_distance)
        relative_distance = np.vectorize(np.uint8)(relative_distance)
        return relative_distance

    def create_cont_ring_image_set(self):
        """Creates a new folder with the newly created ring mask leaves"""
        path_to_folder_lab = os.path.join(
            utils.load_config("PATH", "DATA_DIR"), "Lab_Feb2025_Mask"
        )
        path_to_new_folder = os.path.join(
            utils.load_config("PATH", "DATA_DIR"), "Ring_Mask"
        )
        leaves = os.listdir(path_to_folder_lab)
        for leaf in leaves:
            for side in ["haz", "enves"]:
                leaf_path = os.path.join(path_to_folder_lab, leaf, side)
                for image in os.listdir(leaf_path):
                    leaf_name = image.split(".")[0]
                    lab_mask = open_image.lab_array(leaf_name)
                    cont_ring_mask = img_cleaner.create_cont_ring_array(lab_mask)
                    save_folder = os.path.join(path_to_new_folder, leaf, side)
                    os.makedirs(save_folder, exist_ok=True)
                    save_path = os.path.join(save_folder, image)
                    Image.fromarray(cont_ring_mask).save(save_path)
                    print(f"Saved ring new mask image at {save_path}")


if __name__ == "__main__":
    img_cleaner = ProcessImage()
    from data_mod.open_image import OpenImage

    open_image = OpenImage()
    LEAF = "foliolo10_enves_a12"

    def test_normalise_signal():
        hsi_array = open_image.hsi_array(leaf=LEAF)
        x, y = 150, 100
        spectrogram = hsi_array[x, y, :]
        processed = img_cleaner.normalise_signal(spectrogram)
        channels = list(range(len(spectrogram)))
        plt.plot(
            channels, spectrogram, label="Spectrogram before processing", color="blue"
        )
        plt.plot(channels, processed, label="smooth + snv", color="green")
        plt.legend()
        plt.show()

    def test_relative_distance_mask():
        lab_mask = open_image.lab_array(LEAF)
        dist_mask = img_cleaner.relative_distance_mask(lab_mask)
        plt.imshow(dist_mask)
        plt.colorbar()
        plt.show()

    def test_create_ring_array():
        lab_mask = open_image.lab_array(LEAF)
        ring_mask = img_cleaner.create_class_ring_array(lab_mask)
        plt.imshow(ring_mask)
        plt.colorbar()
        plt.show()

    # test_normalise_signal()
    # test_normalise_signal()
    # test_create_ring_array()
    img_cleaner.create_cont_ring_image_set()
