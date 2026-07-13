import pytest
import torch
import torch.nn as nn
from models.continuous import ODEFunc, ODEBlock
from models.networks import ODENet


def test_shape_consistency() -> None:
    """Ensures that the ODEBlock preserves the input dimensionality."""
    batch_size, dim = 32, 16
    x = torch.randn(batch_size, dim)

    ode_func = ODEFunc(in_features=dim, hidden_dim=32)
    ode_block = ODEBlock(ode_func=ode_func)

    try:
        out = ode_block(x)
    except NotImplementedError:
        pytest.fail("ODEBlock forward pass is not implemented.")

    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_nfe_tracking() -> None:
    """Verifies that Number of Function Evaluations (NFE) strictly increases."""
    dim = 16
    x = torch.randn(1, dim)

    ode_func = ODEFunc(in_features=dim, hidden_dim=32)
    ode_block = ODEBlock(ode_func=ode_func)

    assert ode_func.nfe == 0, "NFE should initialize to 0."

    try:
        _ = ode_block(x)
    except NotImplementedError:
        pytest.fail("ODEBlock forward pass is not implemented.")

    # Solving $\frac{dh}{dt}$ takes at least a few steps, so NFE > 1
    assert ode_func.nfe > 1, f"NFE did not accumulate. NFE is {ode_func.nfe}."


def test_gradient_flow() -> None:
    """Tests the adjoint sensitivity method for valid gradient flow."""
    dim = 16
    x = torch.randn(4, dim, requires_grad=True)

    ode_func = ODEFunc(in_features=dim, hidden_dim=32)
    ode_block = ODEBlock(ode_func=ode_func)

    try:
        out = ode_block(x)
    except NotImplementedError:
        pytest.fail("ODEBlock forward pass is not implemented.")

    loss = out.sum()
    loss.backward()

    has_grads = any(p.grad is not None for p in ode_func.parameters())
    assert has_grads, "Gradients did not flow to ODEFunc."

    assert x.grad is not None, "Gradients did not flow back to the input tensor."


def test_anode_odenet_forward_trajectory_and_backward() -> None:
    """Checks ANODE shapes, augmented trajectories, and gradient flow."""
    batch_size = 5
    data_dim = 2
    hidden_dim = 2
    augment_dim = 2
    num_classes = 2
    expected_ode_dim = hidden_dim + augment_dim

    model = ODENet(
        data_dim=data_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        augment_dim=augment_dim,
    )
    x = torch.randn(batch_size, data_dim)
    targets = torch.tensor([0, 1, 0, 1, 0])

    logits = model(x)
    assert logits.shape == (batch_size, num_classes)

    trajectory = model(x, return_trajectory=True)
    assert trajectory.ndim == 3
    assert trajectory.shape[1] == batch_size
    assert trajectory.shape[-1] == expected_ode_dim

    loss = nn.CrossEntropyLoss()(logits, targets)
    loss.backward()

    has_grads = any(param.grad is not None for param in model.parameters())
    assert has_grads, "Gradients did not flow through the ANODE model."


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available.")
def test_device_agnostic() -> None:
    """Ensures models and tensors stay on the requested GPU device."""
    device = torch.device("cuda:0")
    dim = 16
    x = torch.randn(8, dim).to(device)

    ode_func = ODEFunc(in_features=dim, hidden_dim=32)
    ode_block = ODEBlock(ode_func=ode_func).to(device)

    try:
        out = ode_block(x)
    except NotImplementedError:
        pytest.fail("ODEBlock forward pass is not implemented.")

    assert out.device == device, "Output tensor fell back to CPU."


def test_anode_odenet_with_ode_hidden_dim() -> None:
    """Test ODENet with separate vector-field width under augmentation."""
    batch_size = 4
    data_dim = 2
    num_classes = 2

    model = ODENet(
        data_dim=data_dim,
        hidden_dim=2,
        num_classes=num_classes,
        augment_dim=2,
        ode_hidden_dim=64,
    )

    x = torch.randn(batch_size, data_dim)
    logits = model(x)

    assert logits.shape == (batch_size, num_classes)

    targets = torch.tensor([0, 1, 0, 1])
    loss = nn.CrossEntropyLoss()(logits, targets)
    loss.backward()

    has_grads = any(param.grad is not None for param in model.parameters())
    assert has_grads
