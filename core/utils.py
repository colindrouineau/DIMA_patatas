import yaml

def load_config(
    first_key: str = None,
    second_key: str = None,
    config_path: str = "/home/colind/work/Mines/TR_DIMA/DIMA_code/CONFIG.yaml",
) -> dict[str, object]:
    """
    Returns CONFIG dictionnary or value
    """
    with open(config_path, "r") as f:
        if first_key:
            if second_key:
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

def leaf_training_list(test_leaves):
    """Returns numbers of training leaves (those which are not test)"""
    number_of_leaves = load_config("DATA", "NUMBER_OF_LEAVES")
    all_leaves = list(range(1, number_of_leaves + 1))
    train_leave_numbers = sorted(list(set(all_leaves) - set(test_leaves)))[::-1]
    return train_leave_numbers

