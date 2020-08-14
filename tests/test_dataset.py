import pytest
import torch
from torch.utils.data import DataLoader


@pytest.fixture(scope="module")
def data_shape():
    yield torch.Size((1, 32, 32))


def test_shape(data, data_shape):
    def _check_shape(dataset, msg: str):
        sample, _ = dataset[0]
        assert data_shape == sample.shape, msg

    _check_shape(data.train_data, "train data")
    _check_shape(data.test_data, "test data")


def test_scaling(data):
    def _check_scaling(data, msg: str):
        for sample, _ in data:
            # Values are in range [-1, 1]
            assert 1 >= sample.max(), msg
            assert -1 <= sample.min(), msg

            # Values are not only covering [0,1] or [-1, 0]
            assert torch.any(sample < 0), msg
            assert torch.any(sample > 0), msg

    _check_scaling(data.train_data, "train data")
    _check_scaling(data.test_data, "test data")


def test_augmentation(data):
    def _check_augmentation(data, active: bool, msg: str):
        are_same = []
        for i in range(len(data)):
            sample_1, _ = data[i]
            sample_2, _ = data[i]
            are_same.append(0 == torch.sum(sample_1 - sample_2))

        if active:
            assert not all(are_same), msg
        else:
            assert all(are_same), msg

    _check_augmentation(data.train_data, active=True, msg="train data")
    _check_augmentation(data.test_data, active=False, msg="test data")


def _check_dataloader(data, num_workers: int):
    loader = DataLoader(data, batch_size=4, num_workers=num_workers)
    for _ in loader:
        pass


def test_single_process_dataloader(data):

    _check_dataloader(data.train_data, num_workers=0)
    _check_dataloader(data.test_data, num_workers=0)


def test_multiprocess_dataloader(data):

    _check_dataloader(data.train_data, num_workers=2)
    _check_dataloader(data.test_data, num_workers=2)
