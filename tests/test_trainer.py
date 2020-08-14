import shutil
import tempfile

import numpy as np
import pytest
import scipy.stats
import torch
from torch.utils.data import Subset

from pytest_dl import dataset, model, trainer


@pytest.fixture(
    scope="module",
    params=[
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="No GPU was detected"
            ),
        ),
    ],
)
def f_trainer(request):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = request.param

    # Build dataset with only one batch
    data = dataset.MyMNIST()
    data.train_data = Subset(data.train_data, range(4))
    data.test_data = Subset(data.test_data, range(4))
    vae = model.CNNVAE(data.train_data[0][0].shape, bottleneck_dim=10)
    optim = torch.optim.Adam(vae.parameters())
    log_dir = tempfile.mkdtemp()
    vae_trainer = trainer.Trainer(
        vae,
        data,
        optim,
        batch_size=4,
        device=device,
        log_dir=log_dir,
        num_generated_images=1,
    )

    yield dict(data=data, vae_trainer=vae_trainer)

    shutil.rmtree(log_dir)


@torch.no_grad()
def test_kl_divergence(f_trainer):
    vae_trainer = f_trainer["vae_trainer"]

    mu = np.random.randn(10) * 0.25
    sigma = np.random.randn(10) * 0.1 + 1
    standard_normal_samples = np.random.randn(100000, 10)
    transformed_normal_sample = standard_normal_samples * sigma + mu

    # Calculate empirical pdfs for both distributions
    bins = 1000
    bin_range = [-2, 2]
    expected_kl_div = 0

    for i in range(10):
        standard_normal_dist, _ = np.histogram(
            standard_normal_samples[:, i], bins, bin_range
        )
        transformed_normal_dist, _ = np.histogram(
            transformed_normal_sample[:, i], bins, bin_range
        )
        expected_kl_div += scipy.stats.entropy(
            transformed_normal_dist, standard_normal_dist
        )

    actual_kl_div = vae_trainer._kl_divergence(
        torch.tensor(sigma).log(), torch.tensor(mu)
    )

    assert actual_kl_div.item() == pytest.approx(expected_kl_div, abs=0.05)


def test_overfit_on_one_batch(f_trainer):
    vae_trainer = f_trainer["vae_trainer"]

    # Overfit on single batch
    vae_trainer.train(500)

    # Overfitting a VAE is hard, so we do not choose 0.0 as a goal
    # 30 sum of squared errors would be a deviation of ~0.05 per pixel given a really small KL-Div
    assert vae_trainer.eval() <= 300


def test_logging(f_trainer, mocker):
    vae_trainer = f_trainer["vae_trainer"]

    add_scalar_mock = mocker.patch.object(vae_trainer.summary, "add_scalar")

    vae_trainer.train(1)

    expected_calls = [
        mocker.call("train/recon_loss", mocker.ANY, 0),
        mocker.call("train/kl_div_loss", mocker.ANY, 0),
        mocker.call("train/loss", mocker.ANY, 0),
        mocker.call("test/loss", mocker.ANY, 0),
    ]

    add_scalar_mock.assert_has_calls(expected_calls)
