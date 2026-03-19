import yaml
import ast


def load_config(
    first_key: str = None,
    second_key: str = None,
    third_key: str = None,
    fourth_key: str = None,
    config_path: str = "/home/colind/work/Mines/TR_DIMA/DIMA_code/CONFIG.yaml",
) -> dict[str, object]:
    """
    Returns CONFIG dictionnary or value
    """
    with open(config_path, "r") as f:
        if first_key:
            if second_key:
                if third_key:
                    if fourth_key:
                        return yaml.safe_load(f)[first_key][second_key][third_key][fourth_key]
                    return yaml.safe_load(f)[first_key][second_key][third_key]
                return yaml.safe_load(f)[first_key][second_key]
            return yaml.safe_load(f)[first_key]
        return yaml.safe_load(f)


def sort_images(path_list):
    """
    Sorts leaves or leaf_paths by :
        - key1 = leaf_number
        - key2 = enves, haz
        - key3 = time_state
    """

    def key_sort_func(path):
        leaf = path.split("/")[-1]
        [num, side, _] = leaf.split("_")

        def extract_time_state(leaf):
            leaf = leaf.split(".")[0]
            a_n = leaf.split("_")[-1]
            if len(a_n) == 2:
                return int(a_n[1])
            if len(a_n) == 3:
                return int(a_n[1:])

        time_state = extract_time_state(leaf)
        return (num, side, time_state)

    return sorted(path_list, key=key_sort_func)


def leaf_training_list(not_train_leaves):
    """Returns numbers of training leaves (those which are not test nor validation)"""
    number_of_leaves = load_config("DATA", "NUMBER_OF_LEAVES")
    all_leaves = list(range(1, number_of_leaves + 1))
    train_leave_numbers = sorted(list(set(all_leaves) - set(not_train_leaves)))[::-1]
    return train_leave_numbers


def get_nfeatures_from_name(name):
    """For MLP model name"""
    features_idx = name.find("features")
    n_features = ""
    backward = 1
    while name[features_idx - backward].isdigit():
        n_features += name[features_idx - backward]
        backward += 1
    return int(n_features[::-1])


def get_channels_from_name(name):
    """For tree model name"""
    channels_string = name.split("_")[3].split(".")[0]
    list_string = channels_string.split(":")[1]
    return ast.literal_eval(list_string)

