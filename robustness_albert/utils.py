import json
from pathlib import Path
from typing import Callable, Tuple, Dict, Union, Any

import torch
from datasets import load_dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from transformers import AutoTokenizer

DATASET_SPLITS = ["train", "validation", "test"]
TOKENIZE_COLUMNS = ["input_ids", "attention_mask", "labels"]


def get_dataset(
        task: str,
        model: str,
        sub_task: str = None,
        tokenize: bool = False,
        split: str = None,
        padding: bool = False,
        batched: bool = False,
        return_tokenizer: bool = False,
) -> Union[Callable, Tuple[Callable, Callable]]:
    """Function that returns a dataset and possibly a tokenizer.

    Given the necessary parameters, a dataset is loaded for training and tokenization takes place is specified.
    If tokenize is set to True, the tokenization will take place in this function.
    If split is specified, the returned dataset will only contain samples for the specific split (e.g. only test).
    If return_tokenizer is set to True, then the tokenizer will also be returned.

    Args:
        task (str): Task name that specified which dataset should be loaded (e.g. "glue").
        model (str): Name of the model that will be used in training (e.g. "albert-large-v2").
        sub_task (str): Sub-task name of the dataset if applicable (e.g. in case of "glue"; "sst2").
        tokenize (bool): Boolean that indicates if tokenization should take place or not.
        split (str): Specifies if a specific split of the dataset should only be returned.
        padding (bool): Specifies if padding should be applied while tokenizing.
        batched (bool): Specifies if the tokenization should be batched.
        return_tokenizer (bool): Indicates if the tokenizer should also be returned or not.

    Returns:
        dataset (Dataset): Loaded dataset that will be used.
        tokenizer (Tokenizer, optional): Tokenizer that is/was used for tokenizing the dataset.

    """
    if sub_task:
        dataset = load_dataset(task, sub_task)
    else:
        dataset = load_dataset(task)

    tokenizer = AutoTokenizer.from_pretrained(model)
    if tokenize:
        dataset = dataset.map(
            lambda x: tokenizer(
                x["sentence"],
                padding=padding,
                truncation=True,
            ),
            batched=batched,
        )

    dataset.rename_column_("label", "labels")
    if tokenize:
        dataset.set_format(type='torch', columns=TOKENIZE_COLUMNS)

    if split:
        assert split in DATASET_SPLITS, f"Invalid split, please provide one of the following splits: {DATASET_SPLITS}"
        dataset = dataset[split]

    if return_tokenizer:
        return dataset, tokenizer

    return dataset


def load_config(config_path: str) -> Dict:
    """Function that loads the config file.

    This function loads the config file, given a  path to the configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        config (Dict): Configuration dictionary with all the parameters loaded from the file.
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def save_model(
        model: Any,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        epoch: int,
        folder: str,
        model_name: str
) -> None:
    """Function that saves a model, optimizer, scheduler, and which epoch the training is at.

    Given the folder and model name, this function saves the model and its corresponding optimizer, scheduler, and
    which epoch the training has reached.

    Args:
        model (Model): Model whose weights need to be saved.
        optimizer (Optimizer): Optimizer whose weights need to be saved.
        scheduler (_LRScheduler): Scheduler whose weights need to be saved.
        epoch (int): Epoch where training is right now.
        folder (str): Path to the folder where the model should be saved.
        model_name (str): Name of the model.
    """
    filename = Path(folder) / f"{model_name}_{str(epoch)}.pt"
    Path(folder).mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
    }
    torch.save(checkpoint, filename)


def load_model(
        checkpoint_path: str,
        model: Any,
        optimizer: Optimizer,
        scheduler: _LRScheduler
) -> Tuple[Callable, Optimizer, _LRScheduler, int]:
    """Function that loads a saved model.

    This function loads the weights of the saved model, optimizer, and scheduler, given the path to the checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint that needs to be loaded.
        model (Model): Model whose weights will be loaded (should match the original model).
        optimizer (Optimizer): Optimizer whose weights will be loaded (should match the original optimizer).
        scheduler (_LRScheduler): Scheduler whose weights will be loaded (should match the original scheduler).

    Returns:
        model (Model): Model with loaded weights.
        optimizer (Optimizer): Optimizer with loaded weights.
        scheduler (_LRScheduler): Scheduler with loaded weights.
        epoch (int): Epoch where training ended.

    """
    checkpoint = torch.load(checkpoint_path)
    model_state_dict = checkpoint["model"]
    optimizer_state_dict = checkpoint["optimizer"]
    scheduler_state_dict = checkpoint["scheduler"]
    epoch = checkpoint["epoch"]

    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    scheduler.load_state_dict(scheduler_state_dict)

    return model, optimizer, scheduler, epoch
