import torch
import torch.nn as nn
from models.continuous import ODEFunc, ODEBlock, ConvODEFunc


class ODENet(nn.Module):
    """Full continuous-depth model for classification tasks.

    Two integration regimes (see DEVIATIONS.md):

    * ``use_stem=False`` (toy experiments, Dupont): the ODE integrates in **data
      space** (``d = data_dim``). There is NO learned projection before the flow,
      so the homeomorphism property of the ODE flow applies to the raw inputs --
      this is exactly the regime in which Dupont's toy argument (a NODE cannot
      separate nested regions) holds. Augmentation appends ``augment_dim`` zeros.
    * ``use_stem=True`` (MNIST MLP baseline): a ``Linear(data_dim, hidden_dim) +
      Tanh`` stem projects the (flattened) image before the flow, and the ODE
      integrates in that ``hidden_dim`` representation.
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
        use_stem: bool = True,
        solver_options: dict | None = None,
        head_hidden_dim: int | None = None,
    ) -> None:
        """
        Args:
            data_dim (int): Dimensionality of the input data (e.g., 2 for circles).
            hidden_dim (int): ODE state dimension when ``use_stem=True``. When
                ``use_stem=False`` the ODE state is ``data_dim`` and ``hidden_dim``
                is used only as the fallback vector-field width.
            num_classes (int): Number of output classes.
            solver_type (str): The ODE solver to use.
            atol (float): Absolute error tolerance forwarded to ODEBlock.
            rtol (float): Relative error tolerance forwarded to ODEBlock.
            augment_dim (int): ANODE augmentation dimensions (0 for a standard NODE).
            ode_hidden_dim (Optional[int]): Width of the ODEFunc MLP. Defaults to hidden_dim.
            use_stem (bool): If False, integrate the ODE in data space (no learned
                downsampling stem). Toy experiments set this False; MNIST keeps True.
        """
        super().__init__()

        self.augment_dim: int = augment_dim
        self.use_stem: bool = use_stem

        if use_stem:
            self.downsampling: nn.Module = nn.Sequential(
                nn.Linear(data_dim, hidden_dim), nn.Tanh()
            )
            state_dim = hidden_dim
        else:
            # Data-space integration: identity "stem" keeps utils/plotting code
            # (which calls model.downsampling) working uniformly.
            self.downsampling = nn.Identity()
            state_dim = data_dim

        self.ode_dim: int = state_dim + augment_dim
        vf_hidden_dim: int = hidden_dim if ode_hidden_dim is None else ode_hidden_dim

        self.ode_func = ODEFunc(in_features=self.ode_dim, hidden_dim=vf_hidden_dim)
        self.ode_block = ODEBlock(
            ode_func=self.ode_func,
            solver_type=solver_type,
            atol=atol,
            rtol=rtol,
            options=solver_options,
        )
        # Classifier head applied to the terminal ODE state. Dupont applies a SINGLE
        # LINEAR map L to phi(x); head_hidden_dim=None (default) matches that. A
        # non-None head_hidden_dim builds an MLP head -- used ONLY as a diagnostic
        # contrast to show that a nonlinear head would do the separating work the
        # flow is supposed to do (see DEVIATIONS.md, head row). It is NOT faithful.
        if head_hidden_dim is None:
            self.fc: nn.Module = nn.Linear(self.ode_dim, num_classes)
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.ode_dim, head_hidden_dim),
                nn.ReLU(),
                nn.Linear(head_hidden_dim, num_classes),
            )

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


class EulerDiscretizedODENet(nn.Module):
    r"""Weight-tied Euler discretisation of the ODE-Net (the "discrete baseline").

    IMPORTANT (see DEVIATIONS.md): this is NOT an independent residual architecture.
    It reuses a SINGLE shared ``ODEFunc`` across ``num_layers`` explicit Euler steps

        $h_{t+1} = h_t + \tfrac{1}{L} f_\theta(h_t, t)$,

    which is exactly a fixed-step Euler discretisation of the same ODE the ODE-Net
    integrates adaptively. Weight-tying gives exact parameter parity with the
    ODE-Net, but it means "the discrete model and the ODE-Net reach comparable
    accuracy" is close to tautological -- they are the same vector field, integrated
    two ways. It is a legitimate controlled baseline, not a stand-in for a generic
    ResNet with independent per-layer weights.
    """

    def __init__(
        self, data_dim: int, hidden_dim: int, num_classes: int, num_layers: int = 5
    ):
        super().__init__()
        self.num_layers = num_layers

        self.downsampling = nn.Sequential(nn.Linear(data_dim, hidden_dim), nn.Tanh())

        # Single shared vector field reused across all Euler steps (weight-tied).
        self.layer_func = ODEFunc(in_features=hidden_dim, hidden_dim=hidden_dim)

        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.downsampling(x)

        # Explicit Euler integration with step size dt = 1/num_layers.
        dt = 1.0 / self.num_layers
        for i in range(self.num_layers):
            # Pass a dummy time tensor to match the ODEFunc signature.
            t_dummy = torch.tensor([i * dt], device=x.device, dtype=x.dtype)
            h = h + dt * self.layer_func(t_dummy, h)

        return self.fc(h)


# Backwards-compatible alias (deprecated). The name "DiscreteResNet" overstated
# what this baseline is; use EulerDiscretizedODENet. Kept so any external import
# does not break, but it is not used internally.
DiscreteResNet = EulerDiscretizedODENet
