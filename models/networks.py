import torch
import torch.nn as nn
from models.continuous import ODEFunc, ODEBlock, ConvODEFunc


class ODENet(nn.Module):
    """Full continuous-depth model for classification tasks.

    Args:
        data_dim (int): Dimensionality of the input data (e.g., 2 for circles).
        hidden_dim (int): Dimensionality of the ODE hidden state.
        num_classes (int): Number of output classes (e.g., 2 for binary classification).
        solver_type (str): The ODE solver to use.
        atol (float): Absolute error tolerance forwarded to ODEBlock.
        rtol (float): Relative error tolerance forwarded to ODEBlock.
    """

    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        num_classes: int,
        solver_type: str = "dopri5",
        atol: float = 1e-3,
        rtol: float = 1e-3,
        augment_dim: int = 0,
        ode_hidden_dim: int | None = None,
    ) -> None:
        """Full continuous-depth model for classification tasks with optional augmentation.

        Args:
            data_dim (int): Dimensionality of the input data (e.g., 2 for circles).
            hidden_dim (int): Dimensionality of the ODE hidden state.
            num_classes (int): Number of output classes (e.g., 2 for binary classification).
            solver_type (str): The ODE solver to use.
            atol (float): Absolute error tolerance forwarded to ODEBlock.
            rtol (float): Relative error tolerance forwarded to ODEBlock.
            augment_dim (int): Number of augmentation dimensions for ANODE. 0 for standard NODE.
            ode_hidden_dim (Optional[int]): Width of the ODEFunc MLP. Defaults to hidden_dim for backward compatibility.
        """
        super().__init__()

        self.augment_dim: int = augment_dim
        self.ode_dim: int = hidden_dim + augment_dim

        # ODEFunc MLP width. Defaults to hidden_dim for backward compatibility.
        vf_hidden_dim: int = hidden_dim if ode_hidden_dim is None else ode_hidden_dim

        self.downsampling = nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.Tanh())
        self.ode_func = ODEFunc(in_features=self.ode_dim, hidden_dim=vf_hidden_dim)
        self.ode_block = ODEBlock(
            ode_func=self.ode_func,
            solver_type=solver_type,
            atol=atol,
            rtol=rtol,
        )
        self.fc = nn.Linear(self.ode_dim, num_classes)

    def forward(self, x: torch.Tensor, return_trajectory: bool = False) -> torch.Tensor:
        """Forward pass through the ODENet or ANODE.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, data_dim].
            return_trajectory (bool): If True, return the full ODE trajectory.

        Returns:
            torch.Tensor: Output logits or ODE trajectory.
        """
        self.ode_func.nfe = 0
        h = self.downsampling(x)
        if self.augment_dim > 0:
            zeros = torch.zeros(
                h.shape[0], self.augment_dim, device=h.device, dtype=h.dtype
            )
            h = torch.cat([h, zeros], dim=1)
        h_T = self.ode_block(h, return_trajectory=return_trajectory)

        if return_trajectory:
            return h_T  # We don't pass the full trajectory to the classifier

        return self.fc(h_T)


class ConvODENet(nn.Module):
    """Continuous-depth model for images using Convolutional layers."""

    def __init__(
        self,
        in_channels: int,
        num_filters: int,
        num_classes: int,
        solver_type: str = "dopri5",
    ):
        super().__init__()

        # 1. Map input image (e.g. 1 channel) to feature maps without flattening
        self.downsampling = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )

        # 2. The continuous block using the Convolutional Vector Field
        self.ode_func = ConvODEFunc(num_channels=num_filters)
        self.ode_block = ODEBlock(ode_func=self.ode_func, solver_type=solver_type)

        # 3. Global average pooling to flatten the spatial dimensions, then classify
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_filters, num_classes),
        )

    def forward(self, x: torch.Tensor, return_trajectory: bool = False) -> torch.Tensor:
        self.ode_func.nfe = 0
        h = self.downsampling(x)
        h_T = self.ode_block(h, return_trajectory=return_trajectory)

        if return_trajectory:
            return h_T

        return self.fc(h_T)


class DiscreteResNet(nn.Module):
    """Baseline discrete Residual Network for comparison.

    Uses standard Euler steps: $h_{t+1} = h_t + f(h_t, t)$
    """

    def __init__(
        self, data_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 5
    ):
        super().__init__()
        self.num_layers = num_layers

        self.downsampling = nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.Tanh())

        # Using the exact same vector field architecture for fairness
        self.layer_func = ODEFunc(in_features=hidden_dim, hidden_dim=hidden_dim)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.downsampling(x)

        # Discrete integration (Euler method with step size dt = 1/num_layers)
        dt = 1.0 / self.num_layers
        for i in range(self.num_layers):
            # Pass a dummy time tensor to match the ODEFunc signature
            t_dummy = torch.tensor([i * dt], device=x.device, dtype=x.dtype)
            h = h + dt * self.layer_func(t_dummy, h)

        return self.fc(h)
