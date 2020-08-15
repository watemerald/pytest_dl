import pytest
import torch

from pytest_dl import model


@pytest.fixture(scope="module", params=["cnn", "mlp"])
def net(request):
    if request.param == "cnn":
        return (
            model.CNNVAE(input_shape=(1, 32, 32), bottleneck_dim=16),
            torch.randn(4, 1, 32, 32),
        )
    elif request.param == "mlp":
        return (
            model.MLPVAE(input_shape=(1, 32, 32), bottleneck_dim=16),
            torch.randn(4, 1, 32, 32),
        )
    else:
        raise ValueError("invalid internal test config")


@torch.no_grad()
def test_shape(net):
    net, inputs = net

    outputs = net(inputs)
    assert inputs.shape == outputs.shape


@torch.no_grad()
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU was detected")
def test_device_moving(net):
    net, inputs = net

    net_on_gpu = net.to("cuda:0")
    net_back_on_cpu = net_on_gpu.cpu()

    torch.manual_seed(42)
    outputs_cpu = net(inputs)
    torch.manual_seed(42)
    outputs_gpu = net_on_gpu(inputs)
    torch.manual_seed(42)
    outputs_back_on_cpu = net_back_on_cpu(inputs)

    assert torch.sum(outputs_cpu - outputs_gpu.cpu()) == pytest.approx(0)
    assert torch.sum(outputs_cpu - outputs_back_on_cpu) == pytest.approx(0)


def test_batch_indepependence(net):
    net, inputs = net

    inputs = inputs.clone()
    inputs.requires_grad = True

    # Compute forward pass in eval mode to deactivate batch norm
    net.eval()
    outputs = net(inputs)
    net.train()

    # Mask loss for certain samples in batch
    batch_size = inputs[0].shape[0]
    mask_idx = torch.randint(0, batch_size, ())
    mask = torch.ones_like(outputs)
    mask[mask_idx] = 0
    outputs = outputs * mask

    # Compute backwad pass
    loss = outputs.mean()
    loss.backward()

    # Check if gradient exists and is zero for masked samples
    for i, grad in enumerate(inputs.grad):
        if i == mask_idx:
            assert torch.all(grad == 0).item()
        else:
            assert not torch.all(grad == 0)


def test_all_parameters_updates(net):
    net, inputs = net

    optim = torch.optim.SGD(net.parameters(), lr=0.1)

    outputs = net(inputs)
    loss = outputs.mean()
    loss.backward()
    optim.step()

    for param_name, param in net.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, param_name
            assert torch.sum(param.grad ** 2) != 0.0, param_name
