import pytest

from pytest_dl import dataset


@pytest.fixture(scope="module")
def data():
    yield dataset.MyMNIST()
