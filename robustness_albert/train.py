# Code modified from:
# https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-classification/run_glue_no_trainer.py
# Changed structure of the file, removed unnecessary code (e.g. creating new functions and removing non SST-2
# related code), added comments, and added other necessary code for this research (e.g. incorporating BestEpoch or SWA).
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import random
from typing import Callable, Tuple, Any

import click
import numpy as np
import torch
import wandb
from datasets import load_metric, Metric
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, CyclicLR
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW
from transformers import AlbertForSequenceClassification, set_seed, default_data_collator, \
    DataCollatorWithPadding, get_scheduler

from robustness_albert.lr_schedulers import SWALR
from robustness_albert.utils import get_dataset, load_config, save_model, load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("root")


class BestEpoch:
    """Class to keep track of the best epoch while training.

    Keeps track by comparing the evaluation loss of each epoch. If it is lower than the current best loss, this epoch
    will be considered the best until now, and its evaluation loss will replace the previous best loss.

    Attributes:
        best_epoch (int): Integer indicating the best epoch until now.
        best_loss (float): Float indicating the best loss until now.
        best_accuracy (float): Float indicating the best accuracy until now.

    """
    def __init__(self):
        """Initialize the tracker of the best epoch."""
        self.best_epoch: int = 0
        self.best_loss: float = float("inf")
        self.best_accuracy: float = 0.0

    def update(self, current_loss: float, current_accuracy: float, epoch: int) -> None:
        """Updates the best epoch tracker.

        Takes the evaluation loss and accuracy of the current epoch and compares it with the current best loss.
        If it is lower, updates the current loss, accuracy, and epoch to be the best until now.

        Args:
            current_loss (float): loss of the current epoch.
            current_accuracy (float): accuracy of the current epoch.
            epoch (int): which epoch.
        """
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.best_accuracy = current_accuracy
            self.best_epoch = epoch


def get_optimizer(model: Any, learning_rate: float, weight_decay: float) -> Optimizer:
    """Function that returns the optimizer for training.

    Given the model, learning rate, and weight decay, this function returns the optimizer that can be used while
    training. The model parameters are split into two groups: weight decay and non-weight decay groups, as done in the
    BERT paper.

    Args:
        model (torch.nn.module): Model used for training.
        learning_rate (float): Float that indicates the learning rate.
        weight_decay (float): Float that indicates the weight decay.

    Returns:
        optimizer (Optimizer): optimizer for the training.
    """
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

    return optimizer


def get_dataloader(
        dataset: Dataset, tokenizer: Callable, batch_size: int, padded: bool = False, shuffle: bool = False
) -> DataLoader:
    """Function that returns a dataloader.

    Given a dataset, tokenizer, batch size, and if padding has been applied already or not, a dataloader is returned
    with the appropriate data collator.

    Args:
        dataset (Dataset): Dataset that will be loaded.
        tokenizer (Tokenizer): Tokenizer that will be used if padding has not been applied before.
        batch_size (int): Batch size of the loader.
        padded (bool): Boolean that implies if the data has already been padded or not.
        shuffle (bool): Boolean to indicate if data should be shuffled by dataloader.

    Returns:
        dataloader (DataLoader): Dataloader that loads the dataset.
    """
    # If dataset has been padded already, use default data collator. Else, use collator with padding.
    if padded:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer)

    dataloader = DataLoader(dataset, shuffle=shuffle, collate_fn=data_collator, batch_size=batch_size)
    return dataloader


def train(
        model: Any,
        epoch: int,
        dataloader: DataLoader,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        metric: Metric,
        logging_freq: int,
        max_steps: int,
        device: str,
        enable_swa: bool = False,
        swa_model: AveragedModel = None,
        swa_scheduler: SWALR = None,
) -> None:
    """Function that performs all the steps during the training phase.

    In this function, the entire training phase of an epoch is run. Looping over the dataloader, each batch is fed
    to the model, the loss and metric are tracked/calculated, and the forward and backward pass are done.

    Args:
        model (Model): Model that is being trained.
        epoch (int): Current epoch of experiment.
        dataloader (DataLoader): Object that will load the training data.
        optimizer (Optimizer): Optimizer for training.
        lr_scheduler (_LRScheduler): Learning rate scheduler for the optimizer.
        metric (Metric): Metric that is being tracked.
        logging_freq (int): Frequency of logging the training metrics.
        max_steps (int): Maximum amount of steps to be taken during this epoch.
        device (str): Device on which training will be done.
        enable_swa (bool): Boolean that indicates if Stochastic Weight Averaging should be applied during this epoch.
        swa_model (AveragedModel): Model that does Stochastic Weight Averaging.
        swa_scheduler (SWALR): Learning rate scheduler for Stochastic Weight Averaging.
    """
    model.train()
    logging.info(f" Start training epoch {epoch}")

    if enable_swa:
        logging.info(" SWA is enabled this epoch.")

    accuracies = []
    losses = []
    for step, batch in enumerate(tqdm(dataloader)):

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if step < 5:
            print(batch["input_ids"][0])
            print(dataloader.collate_fn.tokenizer.batch_decode(batch["input_ids"])[0])

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # TODO: Fix for tracking both learning rates necessary?
        if enable_swa and swa_scheduler:
            if isinstance(swa_scheduler, CyclicLR):
                swa_scheduler.step()
            current_lr = swa_scheduler.get_last_lr()[0]
        else:
            lr_scheduler.step()
            current_lr = lr_scheduler.get_last_lr()[0]

        predictions = outputs.logits.argmax(dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
        accuracy = metric.compute()["accuracy"]
        current_step = (epoch * len(dataloader)) + step
        wandb.log(
            {"epoch": epoch, "train_loss": loss, "train_accuracy": accuracy, "learning_rate": current_lr},
            step=current_step
        )

        accuracies.append(accuracy)
        losses.append(loss.detach().cpu().numpy())

        if step % logging_freq == 0:
            logger.info(f" Epoch {epoch}, Step {step}: Loss: {loss}, Accuracy: {accuracy}")

        if current_step == max_steps - 1:
            break

    average_loss = np.mean(losses)
    average_accuracy = np.mean(accuracies)
    wandb.log({"average_train_loss": average_loss, "average_train_accuracy": average_accuracy})
    logger.info(f" Epoch {epoch} average training loss: {average_loss}, accuracy: {average_accuracy}")

    if enable_swa and swa_model and swa_scheduler:
        swa_model.update_parameters(model)
        if isinstance(swa_scheduler, SWALR):
            swa_scheduler.step()
        torch.optim.swa_utils.update_bn(dataloader, swa_model, device=torch.device(device))


def validate(
        model: Any,
        epoch: int,
        dataloader: DataLoader,
        metric: Metric,
        max_steps: int,
        device: str,
        enable_swa: bool = False,
        swa_model: AveragedModel = None,
) -> Tuple[np.float_, float]:
    """Function that performs all the steps during the validation/evaluation phase.

    In this function, the entire evaluation phase of an epoch is run. Looping over the dataloader, each batch is fed
    to the model and the loss and accuracy are tracked.

    Args:
        model (Model): Model that is being trained.:
        epoch (int): Current epoch of experiment.
        dataloader (DataLoader): Object that will load the training data.
        metric (Metric): Metric that is being tracked.
        max_steps (int): Maximum amount of steps to be taken during this epoch.
        device (str): Device on which training will be done.
        enable_swa (bool): Boolean that indicates if SWA should be enabled while evaluating, i.e. the SWA model is used.
        swa_model (AveragedModel): Stochastic Weight Averaging model.

    Returns:
        eval_loss (float): Average loss over the whole validation set.
        eval_accuracy (float): Average accuracy over the whole validation set.
    """
    model.eval()

    if enable_swa and swa_model:
        swa_model.eval()

    with torch.no_grad():
        logger.info(" Starting Evaluation")
        losses = []
        for step, batch in enumerate(tqdm(dataloader)):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            if step < 5:
                print(batch["input_ids"][0])
                print(dataloader.collate_fn.tokenizer.batch_decode(batch["input_ids"])[0])

            if enable_swa and swa_model:
                outputs = swa_model(input_ids, attention_mask, labels=labels)
            else:
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            predictions = outputs.logits.argmax(dim=-1)
            metric.add_batch(
                predictions=predictions,
                references=batch["labels"],
            )
            losses.append(outputs.loss.detach().cpu().numpy())
            current_step = (epoch * len(dataloader)) + step

            if current_step == max_steps - 1:
                break

    eval_loss = np.mean(losses)
    eval_accuracy = metric.compute()["accuracy"]
    logger.info(f" Evaluation {epoch}: Average Loss: {eval_loss}, Average Accuracy: {eval_accuracy}")
    wandb.log({"epoch": epoch, "eval_loss": eval_loss, "eval_accuracy": eval_accuracy})

    return eval_loss, eval_accuracy


@click.command()
@click.option("-c", "--config-path", "config_path", required=True, type=str)
def main(config_path):
    """Function that executes the entire training pipeline.

    This function takes care of loading and processing the config file, initializing the model, dataset, optimizer, and
    other utilities for the entire training job.

    Args:
        config_path (str): path to the config file for the training experiment.
    """
    config = load_config(config_path)

    # Initialize Weights & Biases.
    wandb.init(config=config, project=config["wandb"]["project_name"], name=config["wandb"]["run_name"])
    # Set seeds for reproducibility.
    set_seed(config["pipeline"]["seed"])
    # torch.backends.cudnn.benchmark = False
    # os.environ['PYTHONHASHSEED'] = str(config["pipeline"]["seed"])
    torch.backends.cudnn.deterministic = True

    # Get values from config.
    model_name = config["task"]["model_name"]
    task_name = config["task"]["task_name"]
    sub_task_name = config["task"]["sub_task_name"]
    device = config["pipeline"]["device"]
    swa = "swa" in config

    # Load dataset and dataloaders.
    dataset, tokenizer = get_dataset(
        task_name,
        model_name,
        sub_task=sub_task_name,
        padding=config["processing"]["padding"],
        tokenize=True,
        batched=True,
        return_tokenizer=True
    )
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]
    padding = config["processing"]["padding"]
    train_batch_size = config["pipeline"]["train_batch_size"]
    validation_batch_size = config["pipeline"]["validation_batch_size"]
    train_dataloader = get_dataloader(train_dataset, tokenizer, train_batch_size, padding, shuffle=True)
    validation_dataloader = get_dataloader(validation_dataset, tokenizer, validation_batch_size, padding)

    # Set amount of training steps.
    num_update_steps_per_epoch = len(train_dataloader)
    n_epochs = config["pipeline"]["n_epochs"]
    max_train_steps = n_epochs * num_update_steps_per_epoch
    # If a maximum amount of steps is specified, change the amount of epochs accordingly.
    if "max_train_steps" in config["pipeline"]:
        max_train_steps = config["pipeline"]["max_train_steps"]
        n_epochs = int(np.ceil(max_train_steps / num_update_steps_per_epoch))

    # Load metric, model, optimizer, and learning rate scheduler.
    metric = load_metric(task_name, sub_task_name)
    model = AlbertForSequenceClassification.from_pretrained(model_name)
    optimizer = get_optimizer(model, config["optimizer"]["learning_rate"], config["optimizer"]["weight_decay"])

    lr_scheduler = get_scheduler(
        name=config["optimizer"]["learning_rate_scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["optimizer"]["num_warmup_steps"],
        num_training_steps=max_train_steps,
    )

    swa_model = None
    swa_scheduler = None
    swa_start = None
    swa_schedule_type = None
    if swa:
        swa_model = AveragedModel(model, device=device)
        # Epoch from which SWA should start (count from 0)
        swa_start = config["swa"]["start"]
        swa_schedule_type = config["swa"].get("swa_schedule_type", "constant")

        if swa_schedule_type == "cyclic":
            swa_learning_rate_max = config["swa"]["swa_learning_rate_max"]
            swa_learning_rate_min = config["swa"]["swa_learning_rate_min"]
            step_size_down = config["swa"]["step_size_down"]
            step_size_up = config["swa"]["step_size_up"]
        else:
            anneal_epochs = config["swa"]["anneal_epochs"]
            anneal_strategy = config["swa"]["anneal_strategy"]
            swa_learning_rate = config["swa"]["learning_rate"]
        logging.info(f" Stochastic Weight Averaging is turned on from Epoch {swa_start}")

    # Set everything correctly according to resumption of training or not.
    start_epoch = 0
    if "resume" in config["pipeline"]:
        model, optimizer, scheduler, epoch = load_model(config["pipeline"]["resume"], model, optimizer, lr_scheduler)
        # Start from the next epoch.
        start_epoch = epoch + 1

    model = model.to(device)
    wandb.watch(model, optimizer, log="all", log_freq=10)

    print("\n")
    logger.info(f" Amount training examples: {len(train_dataset)}")
    logger.info(f" Amount validation examples: {len(validation_dataset)}")
    logger.info(f" Amount of epochs: {n_epochs}")
    logger.info(f" Amount optimization steps: {max_train_steps}")
    logger.info(f" Batch size train: {train_batch_size}, validation: {validation_batch_size}")
    print("\n")

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f" Sample {index} of the training set: {train_dataset[index]}.")
    print("\n")

    # Setup best epoch tracker and early stopper if present in config.
    logging_freq = config["pipeline"]["logging_freq"]
    tracker = BestEpoch()

    for epoch in range(start_epoch, n_epochs):
        enable_swa = False

        if swa and epoch == swa_start:
            if swa_schedule_type == "cyclic":
                swa_scheduler = CyclicLR(
                    optimizer,
                    swa_learning_rate_max,
                    swa_learning_rate_min,
                    step_size_down=step_size_down,
                    step_size_up=step_size_up,
                    cycle_momentum=False
                )
            else:
                swa_scheduler = SWALR(
                    optimizer, swa_lr=swa_learning_rate, anneal_epochs=anneal_epochs, anneal_strategy=anneal_strategy
                )

        if swa and epoch >= swa_start:
            enable_swa = True

        train(
            model,
            epoch,
            train_dataloader,
            optimizer,
            lr_scheduler,
            metric,
            logging_freq,
            max_train_steps,
            device,
            enable_swa=enable_swa,
            swa_model=swa_model,
            swa_scheduler=swa_scheduler,
        )
        eval_loss, eval_accuracy = validate(
            model,
            epoch,
            validation_dataloader,
            metric,
            max_train_steps,
            device,
            enable_swa=enable_swa,
            swa_model=swa_model,
        )
        print("\n")

        model_to_save = model
        scheduler_to_save = lr_scheduler
        if enable_swa:
            model_to_save = swa_model
            scheduler_to_save = swa_scheduler
        save_model(
            model_to_save, optimizer, scheduler_to_save, epoch, config["pipeline"]["output_directory"], model_name
        )
        tracker.update(eval_loss, eval_accuracy, epoch)

    logger.info(
        f"Best performance was during epoch {tracker.best_epoch}, with a loss of {tracker.best_loss}, "
        f"and accuracy of {tracker.best_accuracy}"
    )


if __name__ == "__main__":
    main()
