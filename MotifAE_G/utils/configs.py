import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(name)s-%(levelname)s-%(message)s", stream=sys.stdout)

import yaml

# CONSTANT

LOSS_LOG_CONFIG = [
    "spearman_loss",
    "zero_one_loss",
    "l1_loss",
    "total_loss",
    "lr",
    "l1_penalty",
    "zero_one_penalty",
]

# functions

def set_logging_level(level, logger_prefix="SAE"):
    logger_dict = logging.Logger.manager.loggerDict

    for name in logger_dict:
        if name.startswith(logger_prefix):
            logger = logging.getLogger(name)
            logger.setLevel(level)


def get_logger(name):
    return logging.getLogger(name)


logger = get_logger("SAE-Config")


def load_config(config_dir: str):

    if config_dir.endswith("yaml") or config_dir.endswith("yml"):
        with open(config_dir, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        # make input more robust, if the input label is not in a list, change it into a list
        if not config.get("label") is None:
            if isinstance(config.get("label"), str):
                config["label"] = [config["label"]]
        return config

    else:
        raise ValueError("Configuration file must end with yaml or yml")


def check_model_para_config(model):
    """check model parameter setting especially grad info, only used to debug"""
    for i, j in model.named_parameters():
        logger.debug(
            f"Parameter Name: {i}, Parameter Shape: {j.shape}, Parameter Require Grade: {j.requires_grad}"
        )
