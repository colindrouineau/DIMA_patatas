import numpy as np
from PIL import Image
import os
import utils



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

