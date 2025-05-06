import logging
import os
import sys
from pathlib import Path

sys.path.append(Path(__file__).parent)

import click
from dataset import DataPipeline
from model import GatedAutoEncoder, MultipleGatedAutoEncoder
from utils.configs import check_model_para_config, get_logger, load_config, set_logging_level
from utils.trainer import GatedSAETrainer, MultipleGatedSAETrainer

logger = get_logger("SAE-Main")


@click.command()
@click.option("--config_path", "-c", type=str, required=True, help="Config path")
@click.option(
    "--mode", "-m", type=str, required=True, default="train_valid_test", help="Whether to train, valid, test model"
)
@click.option("--test_all_chk", is_flag=True, help="Whether to test on all checkpoint")
@click.option("--new_model_checkpoint", type=str, help="The new model checkpoint")
@click.option("--new_model_checkpoint_base_dir", type=str, help="The base dir to new model checkpoints")
@click.option(
    "--new_test_metadata",
    type=str,
    help="The new test metadata used to replace the org test file in the config, used for flexible new test file",
)
@click.option("--new_test_name", type=str, help="A new test file name to avoid replace the default inference file")
@click.option("--device", type=int, help="Which gpu to load the model, we allow flexibility in device")
def main(
    config_path,
    mode,
    test_all_chk,
    new_model_checkpoint,
    new_model_checkpoint_base_dir,
    new_test_metadata,
    new_test_name,
    device,
):

    # read in config file
    config = load_config(config_path)
    set_logging_level(eval(f"logging.{config['log_mode'] if not config.get('log_mode') is None else 'INFO'}"))

    if not device is None:
        config.update({"device": device})
    if not new_model_checkpoint is None:
        config.update({"new_model_checkpoint": new_model_checkpoint})
    if not new_test_metadata is None:
        config.update({"test_meta_data": new_test_metadata})

    if len(config.get("label", ["score"])) > 1:
        model_class = MultipleGatedAutoEncoder
        trainer_class = MultipleGatedSAETrainer
    else:
        model_class = GatedAutoEncoder
        trainer_class = GatedSAETrainer

    logger.debug(config)

    # set up logging saving env
    os.system(f"mkdir -p {config['log_dir']}/checkpoint")
    os.system(f"mkdir -p {config['log_dir']}/res")

    # init dataloader
    data_pipline = DataPipeline(config)
    ## when we train, we must train, valid and test during training is optional
    if "train" in mode:
        train_loader = data_pipline.get_dataloader(config["train_meta_data"], mode="train")
        if "valid" in mode:
            valid_loader = data_pipline.get_dataloader(config["valid_meta_data"], mode="valid")
        else:
            valid_loader = None
        if "test" in mode:
            test_loader = data_pipline.get_dataloader(config["test_meta_data"], mode="test")
        else:
            test_loader = None
    ## or we only inference on test (only when not in train mode)
    elif "test" in mode:
        test_loader = data_pipline.get_dataloader(config["test_meta_data"], mode="test")
    else:
        raise RuntimeError("Only support train (w/w.o valid/test) / test mode")

    # start training / Inference
    if "train" in mode:

        # init trainer, set up all basic settings of the model
        my_trainer = trainer_class(config=config, model_class=model_class)
        logger.debug("Check SAE para setting")
        check_model_para_config(my_trainer.model)
        logger.debug("Check ESM para setting")
        check_model_para_config(my_trainer.esm_lm)

        my_trainer.train(train_loader, valid_loader, test_loader)

        logger.debug("Check final para change")
        my_trainer.model.check_model_para_change("After Training,")

    # When we only want to do inference
    elif "test" in mode:

        if test_all_chk:
            if new_model_checkpoint_base_dir is None:
                raise RuntimeError(
                    "You may want to specify the test checkpoint base dir using `new_model_checkpoint_base_dir`"
                )
            for checkpoint in os.listdir(new_model_checkpoint_base_dir):
                config.update({"new_model_checkpoint": f"{new_model_checkpoint_base_dir}/{checkpoint}"})
                my_trainer = trainer_class(config=config, model_class=model_class)
                my_trainer.inference(test_loader, "test", new_test_name)

        else:
            if config.get("new_model_checkpoint") is None:
                raise RuntimeError("You may want to specify the test checkpoint")
            my_trainer = trainer_class(config=config, model_class=model_class)
            my_trainer.inference(test_loader, "test", new_test_name)


if __name__ == "__main__":
    main()
