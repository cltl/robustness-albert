import pytest
from robustness_albert.utils import get_dataset, DATASET_SPLITS, TOKENIZE_COLUMNS


@pytest.mark.parametrize(
    "split, amount_split",
    [
        ("train", 67349),
        ("validation", 872),
        ("test", 1821)
    ],
)
def test_get_dataset_split(split, amount_split):
    dataset = get_dataset("glue", "albert-large-v2", sub_task="sst2", tokenize=False, split=split)
    assert dataset.num_rows == amount_split


def test_get_dataset_no_split():
    dataset = get_dataset("glue", "albert-large-v2", sub_task="sst2", tokenize=False)
    assert isinstance(dataset, dict)
    assert len(dataset.keys()) == 3
    for split in DATASET_SPLITS:
        assert split in dataset.keys()


@pytest.mark.parametrize(
    "split, amount_split",
    [
        ("train", 67349),
        ("validation", 872),
        ("test", 1821)
    ],
)
def test_get_dataset_tokenized(split, amount_split):
    dataset = get_dataset("glue", "albert-large-v2", sub_task="sst2", tokenize=True, split=split)
    for column in TOKENIZE_COLUMNS:
        assert column in dataset.column_names
