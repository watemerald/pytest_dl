from enum import Enum

import torch
from typer import Argument, Option, Typer

from pytest_dl import dataset, model, trainer

app = Typer()


class NetworkType(str, Enum):
    mlp = "mlp"
    cnn = "cnn"


@app.command()
def main(
    network_type: NetworkType = Argument(..., help="type of the VAE network"),
    bottleneck_dim: int = Option(
        16, "--bottleneck_dim", "-n", help="size of the VAE bottleneck"
    ),
    lr: float = Option(0.001, "--lr", "-r", help="learning rate for training"),
    batch_size: int = Option(..., "--batch_size", "-b", help="batch size for training"),
    epochs: int = Option(..., "--epochs", "-e", help="epochs to train"),
    device: str = Option(
        "cpu", "--device", "-d", help='device to train on, e.g. "cuda:0"'
    ),
    logdir: str = Option(
        "./results",
        "--logdir",
        "-l",
        help="directory to log the models and event file to",
    ),
):
    """Run the training for a VAE.
    """

    mnist_data = dataset.MyMNIST()

    if network_type == NetworkType.mlp:
        net = model.MLPVAE((1, 32, 32), bottleneck_dim)
    else:
        net = model.CNNVAE((1, 32, 32), bottleneck_dim)

    optim = torch.optim.Adam(net.parameters(), lr)
    vae_trainer = trainer.Trainer(net, mnist_data, optim, batch_size, device, logdir)
    vae_trainer.train(epochs)


if __name__ == "__main__":
    app()
