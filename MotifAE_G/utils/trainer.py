import copy
import os
import time
import uuid

import esm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils.configs import LOSS_LOG_CONFIG, get_logger

logger = get_logger("SAE-Trainer")
import utils.loss as loss_f


class GatedSAETrainer:
    def __init__(self, config: dict, model_class: nn.Module):

        # load parameters from the config
        ## basic settings
        self.device = f"cuda:{config['device']}" if not config.get("device") is None else "cpu"
        self.log_dir = config["log_dir"]
        self.log_step = config["log_step"]
        self.save_checkpoint_step = config["save_checkpoint_step"]

        ## training settings
        self.lr = config["lr"]
        self.warmup_steps = config["warmup_steps"]
        self.max_epoch = config["n_epoch"]

        ## loss settings
        self.zero_one_penalty = config["zero_one_penalty"]
        self.zero_one_annealing_steps = config["zero_one_annealing_steps"]
        self.current_zero_one_penalty = 0 if self.zero_one_annealing_steps > 0 else self.zero_one_penalty
        self.l1_penalty = config["l1_penalty"]
        self.l1_annealing_steps = config["l1_annealing_steps"]
        self.current_l1_penalty = 0 if self.l1_annealing_steps > 0 else self.l1_penalty
        self.rank_reg_strength = config["rank_reg_strength"]
        self.rank_reg_method = config["rank_reg_method"]

        # Set random seeds
        if config["seed"] is not None:
            torch.manual_seed(config["seed"])
            torch.cuda.manual_seed_all(config["seed"])

        # Initialize model
        para_name_map_dict = config.get("para_name_map_dict", {})
        if not config.get("new_model_checkpoint") is None:
            checkpoint_path = config["new_model_checkpoint"]
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            self.model = model_class.from_pretrained(
                checkpoint=checkpoint["model_state_dict"],
                para_name_map_dict=para_name_map_dict,
                device=self.device,
            )
            history_step_num = int(checkpoint_path.split("/")[-1].split("_")[0])
            logger.info(f"Loaded GatedSAE Model from {checkpoint_path}. Continue from step {history_step_num}.")
        elif not config.get("std_model_checkpoint") is None:
            checkpoint_path = config["std_model_checkpoint"]
            checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
            if not checkpoint.get("model_state_dict") is None:
                chk = checkpoint["model_state_dict"]
            else:
                chk = checkpoint
            self.model = model_class.from_pretrained(
                checkpoint=chk,
                para_name_map_dict=para_name_map_dict,
                device=self.device,
            )
            history_step_num = 0
            logger.info(f"Loaded Standard SAE Model from {checkpoint_path}. From scratch.")
        else:
            raise RuntimeError("Don't support train from scratch")
        self.model.temperature = 1 if config.get("temperature") is None else config.get("temperature")

        ## load esm model and freeze parameters
        esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_lm = esm_model.lm_head
        self.esm_lm.to(self.device)
        logger.info("Finished Loading ESM2 Model")

        ## freeze parameters for SAE and ESM
        freeze_para = config.get("freeze_para", [])
        for n, para in self.model.named_parameters():
            if n in freeze_para:
                para.requires_grad = False
        for para in esm_model.lm_head.parameters():
            para.requires_grad = False

        # Initialize optimizer and schedular, only update SAE parameter
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.schedular = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda step: min(step / self.warmup_steps, 1.0)
        )
        ## if load from a trained model, we need to continue training and load the optimizer and schedular as well
        if history_step_num != 0:
            logger.info(f"Optimizer continue from step {history_step_num}.")
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.schedular.load_state_dict(checkpoint["schedular_state_dict"])

        ## config the step we are at, this is the hitory step since we start training this model
        self.step = history_step_num

        ## loss log config
        self.loss_log_config = config.get("loss_log") if not config.get("loss_log") is None else LOSS_LOG_CONFIG

    @staticmethod
    def update_penalty(step, annealing_steps, final_penalty):
        """Update penalty according to annealing schedule."""
        if step < annealing_steps:
            current_penalty = final_penalty * (step / annealing_steps)
        else:
            current_penalty = final_penalty
        return current_penalty

    @property
    def current_lr(self):
        """Get current optimizer learning rate."""
        return self.optimizer.param_groups[0]["lr"]

    def get_loss(self, x, y, pos, wt, alt, logging=False, return_pred=False):
        """Compute loss for current batch. We also get the pred if necessary. The return value always contain the true loss, the loss log and the pred (but if some are unnecessary, we will not construct that variable)"""
        # For the original batch construction, we have
        ## x shape: [batch, L, 1280]
        ## x_hat shape: [batch, L, 1280]
        ## output shape: [batch, L, 33]
        # For the current batch construction, we have
        ## x shape: [batch, 1280]

        # y shape: [batch]
        # pos shape: [batch]
        # alt shape: [batch]
        batchsize = x.shape[0]
        x_hat = self.model(x)
        output = self.esm_lm(x_hat)
        logits = torch.log_softmax(output, dim=-1)
        gate_sigmoid = torch.sigmoid(self.model.gate / self.model.temperature)

        # rank loss (on the output)
        ## pred shape:  [batch]
        # pred = logits[range(batchsize), pos, [self.alphabet.tok_to_idx[i] for i in alt]]  # for the org batch construction
        pred = (
            logits[range(batchsize), [self.alphabet.tok_to_idx[i] for i in alt]]
            - logits[range(batchsize), [self.alphabet.tok_to_idx[i] for i in wt]]
        )
        spearman_loss = loss_f.spearman_loss(
            pred, y[:, 0], reg_strength=self.rank_reg_strength, reg_method=self.rank_reg_method
        )

        # zero-one loss (on the gate)
        zero_one_loss = loss_f.zero_one_loss(gate_sigmoid)

        # l1 loss (on the gate)
        l1_loss = loss_f.sparsity_loss(gate_sigmoid)

        # final loss
        total_loss = (
            spearman_loss + self.current_zero_one_penalty * zero_one_loss + self.current_l1_penalty * l1_loss
        )

        logger.debug(
            f"Step {self.step}, spearman loss: {spearman_loss}, l1 loss: {l1_loss}, zero-one loss: {zero_one_loss}"
        )

        # return logging and pred
        if logging:
            all_log = {
                "spearman_loss": spearman_loss.item(),
                "zero_one_loss": zero_one_loss.item(),
                "l1_loss": l1_loss.item(),
                "total_loss": total_loss.item(),
                "lr": self.current_lr,
                "l1_penalty": self.current_l1_penalty,
                "zero_one_penalty": self.current_zero_one_penalty,
            }

            loss_log = pd.Series({k: all_log[k] for k in self.loss_log_config})
        else:
            loss_log = None

        if return_pred:
            return total_loss, loss_log, pred
        else:
            return total_loss, loss_log, None

    def single_step_update(self, x, y, pos, wt, alt):
        """
        Perform single training step.

        Args:
            step: Current training step
            x: Batch of input esm2 feature
            y: Batch of final label
            pos: Batch of mut position
            alt: Batch of mut aa
        """
        if self.zero_one_annealing_steps > 0:
            self.current_zero_one_penalty = self.update_penalty(
                self.step, self.zero_one_annealing_steps, self.zero_one_penalty
            )
        if self.l1_annealing_steps > 0:
            self.current_l1_penalty = self.update_penalty(self.step, self.l1_annealing_steps, self.l1_penalty)

        # Compute and apply gradients
        self.optimizer.zero_grad()

        loss, loss_log, _ = self.get_loss(x, y, pos, wt, alt, logging=True, return_pred=False)

        loss.backward()

        # Clip gradients by value
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.5)

        self.optimizer.step()
        self.schedular.step()

        return loss_log.to_frame().T

    def train(self, train_dataloader, valid_loader, test_loader):
        logger.info("Start Training")
        logger.info(f"Expected Training Step: {self.step + self.max_epoch * len(train_dataloader)}")
        logger.info(f"Total Batch Num: {len(train_dataloader)}")

        # set up some logging info
        self.step += 1
        all_loss_log = pd.DataFrame(columns=self.loss_log_config)
        start_time = time.time()

        if os.path.exists(f"{self.log_dir}/training_log.csv"):
            log = pd.read_csv(f"{self.log_dir}/training_log.csv", index_col=0)
        else:
            log = pd.DataFrame()
            log.index.name = "step"

        for epoch in range(self.max_epoch):
            for x, y, pos, wt, alt, _ in train_dataloader:

                self.model.train()
                self.esm_lm.train()
                x, y = x.to(self.device), y.to(self.device)

                step_loss_log = self.single_step_update(x, y, pos, wt, alt)
                # keep track of all information across training batches
                all_loss_log = pd.concat([all_loss_log, step_loss_log], axis=0, ignore_index=True)

                # saving logs
                if self.step % self.log_step == 0:
                    # training time
                    log.loc[self.step, "time"] = time.time() - start_time
                    start_time = time.time()

                    # loss log info, here we also mean the lr, penalty across the logging batches
                    log.loc[self.step, self.loss_log_config] = all_loss_log.mean()
                    all_loss_log = pd.DataFrame(columns=self.loss_log_config)

                    # sparsity metric / 0-1 metric
                    gate_sigmoid = torch.sigmoid(self.model.gate / self.model.temperature)
                    log.loc[self.step, "closed_feature_percentage"] = (
                        (gate_sigmoid <= 0.5).detach().cpu().numpy().mean()
                    )

                    # save the logging file
                    log.to_csv(f"{self.log_dir}/training_log.csv")
                    logger.info(f"\n{log.loc[self.step]}")

                    # valid and test
                    if not valid_loader is None:
                        self.inference(valid_loader, inference_type="valid")
                    if not test_loader is None:
                        self.inference(test_loader, inference_type="test")

                if self.step % self.save_checkpoint_step == 0:
                    # here we save model weights and the optimizer and schedular info
                    checkpoint = {
                        "model_state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "schedular_state_dict": self.schedular.state_dict(),
                    }
                    save_file = f"{self.log_dir}/checkpoint/{self.step}_step.pt"
                    if os.path.exists(save_file):
                        random_id = uuid.uuid4().hex[:8]
                        logger.warning(f"Already have checkpoint at {save_file}. Save extra uuid {random_id}")
                        save_file = f"{self.log_dir}/checkpoint/{self.step}_step_{random_id}.pt"
                    torch.save(checkpoint, save_file)

                self.step += 1

    def inference(self, dataloader, inference_type, inference_file_name=None):
        # only need this to specify new inference file name, add flexibility
        if inference_file_name is None:
            inference_file_name = inference_type

        inference_res = pd.DataFrame()

        self.model.eval()
        self.esm_lm.eval()

        logger.info(f"Step {self.step}. Inference in {inference_type} mode.")

        start_time = time.time()
        with torch.no_grad():
            for x, y, pos, wt, alt, sample_idx in dataloader:
                x, y = x.to(self.device), y.to(self.device)

                # res log
                res_log = {
                    "relative_pos_py_idx": pos,
                    "wt": wt,
                    "alt": alt,
                    "sample_idx": sample_idx,
                    "score": y[:, 0].detach().cpu(),
                }
                if inference_type == "valid":
                    _, loss_log, pred = self.get_loss(x, y, pos, wt, alt, logging=True, return_pred=True)
                    res_log.update({"pred": pred.detach().cpu()})
                    res_log.update(loss_log)
                elif inference_type == "test":
                    _, _, pred = self.get_loss(x, y, pos, wt, alt, logging=False, return_pred=True)
                    res_log.update({"pred": pred.detach().cpu()})
                else:
                    raise RuntimeError("Not supported inference type. Only valid/test")

                batch_res = pd.DataFrame(res_log)
                inference_res = pd.concat([inference_res, batch_res], axis=0, ignore_index=True)

        inference_res.set_index("sample_idx", drop=True).to_csv(
            f"{self.log_dir}/res/{inference_file_name}_infer_res_step_{self.step}.csv"
        )

        end_time = time.time()
        logger.info(f"Inference time: {end_time - start_time:.2f} sec")


class MultipleGatedSAETrainer(GatedSAETrainer):

    def __init__(self, config, model_class):
        super().__init__(config, model_class)

        self.label_name = config.get("label", ["score"])
        self.intersection_penalty = config.get("intersection_penalty", 1e-3)

    def get_loss(self, x, y, pos, wt, alt, logging=False, return_pred=False):
        """Compute loss for current batch. We also get the pred if necessary. The return value always contain the true loss, the loss log and the pred (but if some are unnecessary, we will not construct that variable)"""
        # For the original batch construction, we have
        ## x shape: [batch, L, 1280]
        ## x_hat shape: [batch, L, 1280]
        ## output shape: [batch, L, 33]
        # For the current batch construction, we have
        ## x shape: [batch, 1280]

        # y shape: [batch]
        # pos shape: [batch]
        # alt shape: [batch]
        batchsize = x.shape[0]

        spearman_loss_list = []
        zero_one_loss_list = []
        l1_loss_list = []
        pred_list = []
        mask_list = []

        x_hat_list = self.model(x)

        for label_idx in range(len(self.label_name)):
            output = self.esm_lm(x_hat_list[label_idx])
            logits = torch.log_softmax(output, dim=-1)
            gate_sigmoid = torch.sigmoid(self.model.gate_list[label_idx] / self.model.temperature)
            gate_mask = (gate_sigmoid > 0.5).int()

            # rank loss (on the output)
            ## pred shape:  [batch]
            # pred = logits[range(batchsize), pos, [self.alphabet.tok_to_idx[i] for i in alt]]  # for the org batch construction
            pred = (
                logits[range(batchsize), [self.alphabet.tok_to_idx[i] for i in alt]]
                - logits[range(batchsize), [self.alphabet.tok_to_idx[i] for i in wt]]
            )
            spearman_loss = loss_f.spearman_loss(
                pred, y[:, label_idx], reg_strength=self.rank_reg_strength, reg_method=self.rank_reg_method
            )

            # zero-one loss (on the gate)
            zero_one_loss = loss_f.zero_one_loss(gate_sigmoid)

            # l1 loss (on the gate)
            l1_loss = loss_f.sparsity_loss(gate_sigmoid)

            spearman_loss_list.append(spearman_loss)
            zero_one_loss_list.append(zero_one_loss)
            l1_loss_list.append(l1_loss)
            pred_list.append(pred)
            mask_list.append(gate_mask)

        intersection = torch.ones(mask_list[0].shape, dtype=int, device=self.device)
        spearman_loss_all = 0
        zero_one_loss_all = 0
        l1_loss_all = 0
        for _ in range(len(self.label_name)):
            intersection *= mask_list[_]
            spearman_loss_all += spearman_loss_list[_]
            zero_one_loss_all += zero_one_loss_list[_]
            l1_loss_all += l1_loss_list[_]
        intersection_loss = torch.sum(intersection)
        # final loss
        total_loss = (
            spearman_loss_all
            + self.current_zero_one_penalty * zero_one_loss_all
            + self.current_l1_penalty * l1_loss_all
            + self.intersection_penalty * intersection_loss
        )

        # return logging and pred
        if logging:
            all_log = {
                "spearman_loss_all": spearman_loss_all.item(),
                "zero_one_loss_all": zero_one_loss_all.item(),
                "l1_loss_all": l1_loss_all.item(),
                "intersection_loss": intersection_loss.item(),
                "total_loss": total_loss.item(),
                "lr": self.current_lr,
                "l1_penalty": self.current_l1_penalty,
                "zero_one_penalty": self.current_zero_one_penalty,
                "intersection_penalty": self.intersection_penalty,
            }
            for idx in range(len(self.label_name)):
                all_log.update(
                    {
                        f"spearman_loss_label{idx}": spearman_loss_list[idx].item(),
                        f"zero_one_loss_label{idx}": zero_one_loss_list[idx].item(),
                        f"l1_loss_label{idx}": l1_loss_list[idx].item(),
                    }
                )

            loss_log = pd.Series({k: all_log[k] for k in self.loss_log_config})
        else:
            loss_log = None

        if return_pred:
            return total_loss, loss_log, torch.stack(pred_list).T
        else:
            return total_loss, loss_log, None

    def train(self, train_dataloader, valid_loader, test_loader):
        logger.info("Start Training")
        logger.info(f"Expected Training Step: {self.step + self.max_epoch * len(train_dataloader)}")
        logger.info(f"Total Batch Num: {len(train_dataloader)}")

        # set up some logging info
        self.step += 1
        all_loss_log = pd.DataFrame(columns=self.loss_log_config)
        start_time = time.time()

        if os.path.exists(f"{self.log_dir}/training_log.csv"):
            log = pd.read_csv(f"{self.log_dir}/training_log.csv", index_col=0)
        else:
            log = pd.DataFrame()
            log.index.name = "step"

        for epoch in range(self.max_epoch):
            for x, y, pos, wt, alt, _ in train_dataloader:

                self.model.train()
                self.esm_lm.train()

                x, y = x.to(self.device), y.to(self.device)

                step_loss_log = self.single_step_update(x, y, pos, wt, alt)
                # keep track of all information across training batches
                all_loss_log = pd.concat([all_loss_log, step_loss_log], axis=0, ignore_index=True)

                # saving logs
                if self.step % self.log_step == 0:
                    # training time
                    log.loc[self.step, "time"] = time.time() - start_time
                    start_time = time.time()

                    # loss log info, here we also mean the lr, penalty across the logging batches
                    log.loc[self.step, self.loss_log_config] = all_loss_log.mean()
                    all_loss_log = pd.DataFrame(columns=self.loss_log_config)

                    # sparsity metric / 0-1 metric
                    for label_idx, l in enumerate(self.label_name):
                        gate_sigmoid = torch.sigmoid(self.model.gate_list[label_idx] / self.model.temperature)
                        log.loc[self.step, f"closed_feature_percentage_{l}"] = (
                            (gate_sigmoid <= 0.5).detach().cpu().numpy().mean()
                        )

                    # save the logging file
                    log.to_csv(f"{self.log_dir}/training_log.csv")
                    logger.info(f"\n{log.loc[self.step]}")

                    # valid and test
                    if not valid_loader is None:
                        self.inference(valid_loader, inference_type="valid")
                    if not test_loader is None:
                        self.inference(test_loader, inference_type="test")

                if self.step % self.save_checkpoint_step == 0:
                    # here we save model weights and the optimizer and schedular info
                    checkpoint = {
                        "model_state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "schedular_state_dict": self.schedular.state_dict(),
                    }
                    save_file = f"{self.log_dir}/checkpoint/{self.step}_step.pt"
                    if os.path.exists(save_file):
                        random_id = uuid.uuid4().hex[:8]
                        logger.warning(f"Already have checkpoint at {save_file}. Save extra uuid {random_id}")
                        save_file = f"{self.log_dir}/checkpoint/{self.step}_step_{random_id}.pt"
                    torch.save(checkpoint, save_file)

                self.step += 1

    def inference(self, dataloader, inference_type, inference_file_name=None):
        # only need this to specify new inference file name, add flexibility
        if inference_file_name is None:
            inference_file_name = inference_type

        inference_res = pd.DataFrame()

        self.model.eval()
        self.esm_lm.eval()

        logger.info(f"Step {self.step}. Inference in {inference_type} mode.")

        start_time = time.time()
        with torch.no_grad():
            for x, y, pos, wt, alt, sample_idx in dataloader:
                x, y = x.to(self.device), y.to(self.device)

                # res log
                res_log = {
                    "relative_pos_py_idx": pos,
                    "wt": wt,
                    "alt": alt,
                    "sample_idx": sample_idx,
                }
                for label_idx, l in enumerate(self.label_name):
                    res_log.update({l: y[:, label_idx].detach().cpu()})

                if inference_type == "valid":
                    _, loss_log, pred = self.get_loss(x, y, pos, wt, alt, logging=True, return_pred=True)
                    for label_idx, l in enumerate(self.label_name):
                        res_log.update({f"pred_{l}": pred[:, label_idx].detach().cpu()})
                    res_log.update(loss_log)
                elif inference_type == "test":
                    _, _, pred = self.get_loss(x, y, pos, wt, alt, logging=False, return_pred=True)
                    for label_idx, l in enumerate(self.label_name):
                        res_log.update({f"pred_{l}": pred[:, label_idx].detach().cpu()})
                else:
                    raise RuntimeError("Not supported inference type. Only valid/test")

                batch_res = pd.DataFrame(res_log)
                inference_res = pd.concat([inference_res, batch_res], axis=0, ignore_index=True)

        inference_res.set_index("sample_idx", drop=True).to_csv(
            f"{self.log_dir}/res/{inference_file_name}_infer_res_step_{self.step}.csv"
        )

        end_time = time.time()
        logger.info(f"Inference time: {end_time - start_time:.2f} sec")
