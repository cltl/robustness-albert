from typing import List

import pytest
from tqdm import tqdm
from transformers import set_seed

from robustness_albert.train import BestEpoch, get_dataloader
from robustness_albert.utils import get_dataset


@pytest.mark.parametrize(
    "losses, accuracies, best_epoch, best_loss, best_accuracy",
    [
        ([0.6, 0.5, 0.4, 0.3, 0.2], [0.6, 0.7, 0.8, 0.9, 1.0], 4, 0.2, 1.0),
        ([0.3, 0.5, 0.1, 0.3], [0.6, 0.4, 0.9, 1.0], 2, 0.1, 0.9)
    ],
)
def test_epoch_tracker(
        losses: List[float],
        accuracies: List[float],
        best_epoch: int,
        best_loss: float,
        best_accuracy: float
):
    epoch_tracker = BestEpoch()

    for i, (loss, accuracy) in enumerate(zip(losses, accuracies)):
        epoch_tracker.update(loss, accuracy, i)

    assert epoch_tracker.best_epoch == best_epoch
    assert epoch_tracker.best_loss == best_loss
    assert epoch_tracker.best_accuracy == best_accuracy


@pytest.mark.parametrize(
    "train_batch_size, shuffle",
    [
        (32, True),
        (64, False),
    ],
)
def test_dataloader(train_batch_size: int, shuffle: bool, n_epochs: int = 2, seed: int = 0):
    set_seed(seed)
    dataset, tokenizer = get_dataset(
        "glue",
        "albert-large-v2",
        sub_task="sst2",
        padding=False,
        tokenize=True,
        batched=True,
        return_tokenizer=True
    )
    train_dataset = dataset["train"]
    train_dataloader = get_dataloader(train_dataset, tokenizer, train_batch_size, padded=False, shuffle=shuffle)

    first_sample_per_epoch = []
    all_previous_data = []
    for i in range(n_epochs):
        print(f"Epoch {i}")

        first_data = []
        all_data = []
        n_samples = 0
        for step, batch in enumerate(tqdm(train_dataloader)):
            input_ids = batch["input_ids"]
            # We do the tests with the first sample of the batch to reduce the complexity.
            first_data.append(input_ids[0])
            all_data.extend(input_ids)
            n_samples += len(input_ids)

            if step == 0:
                first_sample_per_epoch.append(input_ids[0])

        # Check if no data sample is skipped.
        assert (n_samples // train_batch_size) + 1 == len(train_dataloader)
        # Check if there are no duplicate data points made by the dataloader.
        assert len(set(first_data)) == len(first_data)
        assert len(first_data) == step + 1
        assert len(set(all_data)) == len(all_data)

        first_data = [sample.tolist() for sample in first_data]

        if i != 0:
            # Check if the previous epoch contained the same data.
            assert len(all_previous_data) > 0
            assert len(all_previous_data) == len(first_data)

            if shuffle:
                # This list should not contain the same first samples every batch.
                assert all_previous_data != first_data
            else:
                assert all_previous_data == first_data

        all_previous_data = first_data

    if shuffle:
        # Check if data is actually shuffled by verifying if the first sample each epoch is not the same.
        assert len(set(first_sample_per_epoch)) == len(first_sample_per_epoch)
