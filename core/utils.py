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