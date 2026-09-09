from scipy.ndimage import distance_transform_edt
import numpy as np
import utils
import os
from PIL import Image


class CreateMask:
    def __init__(self):
        pass

    def relative_distance_mask(self, lab_mask):
        """
        Compute the distance from each point to the closest point of the other class (0 or 1), ignoring -1. T

        Args:
            arr: 2D array of 0, 255, 200 (output of `OpenImage().lab_array`).

        Returns:
            distance_map: 2D array where each element is the distance to the closest point of the other class.
                          Encoding method: distance to closest sane point is increased by 128 to differentiate with the distance to sick
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
        distance_map[mask_1] = 128 + distance_to_0[mask_1]
        distance_map[arr == -1] = 0  # if out of the leaf

        return distance_map.astype("uint8")

    def create_relativedist_mask(self):
        """Creates a new folder with the newly created ring mask leaves"""
        path_to_folder_lab = os.path.join(
            utils.load_config("PATH", "DATA_DIR"), "Lab_Feb2025_Mask"
        )
        path_to_new_folder = os.path.join(
            utils.load_config("PATH", "DATA_DIR"), "Mask_RelDist"
        )
        leaves = os.listdir(path_to_folder_lab)
        for leaf in leaves:
            for side in ["haz", "enves"]:
                leaf_path = os.path.join(path_to_folder_lab, leaf, side)
                for image in os.listdir(leaf_path):
                    leaf_name = image.split(".")[0]
                    lab_mask = open_image.lab_array(leaf_name)
                    rel_dist_mask = img_cleaner.relative_distance_mask(lab_mask)
                    save_folder = os.path.join(path_to_new_folder, leaf, side)
                    os.makedirs(save_folder, exist_ok=True)
                    save_path = os.path.join(save_folder, image)
                    Image.fromarray(rel_dist_mask).save(save_path)
                    print(f"Saved relative distance new mask image at {save_path}")

    def create_temporal_mask(self):
        """Creates a new folder with the newly created temporal mask leaves

        The mask is such that the pixel value is 128 if the pixel is sick in the next image and wasn't sick in the current image,
        and is 128 + (number of images before contamination (can be negative)) for all pixels.
        The last image of the sequence won't be labeled.
        """

        path_to_folder_lab = os.path.join(
            utils.load_config("PATH", "DATA_DIR"), "Lab_Feb2025_Mask"
        )
        path_to_new_folder = os.path.join(
            utils.load_config("PATH", "DATA_DIR"), "Temporal_Mask"
        )
        leaves = utils.sort_leaves(os.listdir(path_to_folder_lab))
        for leaf in leaves:
            for side in ["haz", "enves"]:
                leaf_path = os.path.join(path_to_folder_lab, leaf, side)

                # load masks
                leaf_sequence = []
                image_paths = utils.sort_images(os.listdir(leaf_path))
                for image in image_paths:
                    leaf_name = image.split(".")[0]
                    lab_mask = open_image.lab_array(leaf_name)
                    leaf_sequence.append(lab_mask)

                # compute temporal masks. binary
                temporal_mask_sequence = []
                for i in range(len(leaf_sequence) - 1):
                    new_sick = leaf_sequence[i] - leaf_sequence[i + 1]
                    new_sick = np.where(new_sick == 55, np.uint8(128), np.uint8(0))
                    temporal_mask_sequence.append(new_sick)
                # add gradient
                grad_temp_seq = [
                    np.copy(temporal_mask_sequence[i])
                    for i in range(len(temporal_mask_sequence))
                ]
                for i in range(len(leaf_sequence) - 1):
                    for j in range(len(leaf_sequence) - 1):
                        grad_temp_seq[i] += np.where(
                            grad_temp_seq[i] == 0,
                            (j - i + 128) * (temporal_mask_sequence[j] // 128),
                            0,
                        )

                # save temporal masks
                for i, image in enumerate(image_paths):
                    if i == len(grad_temp_seq):
                        break
                    leaf_name = image.split(".")[0]
                    save_folder = os.path.join(path_to_new_folder, leaf, side)
                    os.makedirs(save_folder, exist_ok=True)
                    save_path = os.path.join(save_folder, image)
                    Image.fromarray(grad_temp_seq[i]).save(save_path)
                    print(f"Saved temporal new mask image at {save_path}")


if __name__ == "__main__":
    img_cleaner = CreateMask()
    from data_mod.open_image import OpenImage

    open_image = OpenImage()
    LEAF = "foliolo10_enves_a12"

    # test_normalise_signal()
    # test_normalise_signal()
    # test_create_ring_array()
    # img_cleaner.create_cont_ring_image_set()
    img_cleaner.create_temporal_mask()
    img_cleaner.create_relativedist_mask()
