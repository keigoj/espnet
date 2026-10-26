import math
from typing import Optional

import torch
from typeguard import typechecked


class UniformSinkhornDivergence(torch.nn.Module):
    """POT-backed Sinkhorn divergence, with uniform masses by default.

    POT selects its PyTorch backend from the input tensors and computes the
    transport plan with its log-domain Sinkhorn solver.  ``grad="envelope"``
    preserves gradients to the point coordinates without retaining all solver
    iterations in the autograd graph.
    """

    @typechecked
    def __init__(
        self,
        epsilon: float = 1.0,
        num_iterations: int = 20,
        normalize_by_dimension: bool = True,
        stop_threshold: float = 1.0e-5,
    ):
        super().__init__()
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive: {epsilon}")
        if num_iterations <= 0:
            raise ValueError(f"num_iterations must be positive: {num_iterations}")
        if stop_threshold <= 0.0:
            raise ValueError(f"stop_threshold must be positive: {stop_threshold}")
        self.epsilon = epsilon
        self.num_iterations = num_iterations
        self.normalize_by_dimension = normalize_by_dimension
        self.stop_threshold = stop_threshold
        self.register_buffer(
            "reference_self_cost", torch.tensor(float("nan")), persistent=False
        )

    @staticmethod
    def _normalize_mass(
        points: torch.Tensor,
        mass: Optional[torch.Tensor],
        name: str,
    ) -> torch.Tensor:
        if mass is None:
            return points.new_full((points.size(0),), 1.0 / points.size(0)).float()
        if mass.dim() != 1 or mass.numel() != points.size(0):
            raise ValueError(
                f"{name} must have shape ({points.size(0)},), got {tuple(mass.shape)}"
            )
        normalized = mass.to(device=points.device, dtype=torch.float32)
        if not torch.isfinite(normalized).all():
            raise ValueError(f"{name} must contain only finite values")
        if not (normalized > 0.0).all():
            raise ValueError(f"{name} values must all be positive")
        return normalized / normalized.sum()

    def _regularized_ot(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_mass: torch.Tensor,
        target_mass: torch.Tensor,
    ) -> torch.Tensor:
        try:
            import ot
        except ImportError as error:
            raise ImportError(
                "POT>=0.9.7,<0.10 is required for the optimal-transport loss. "
                "Install ESPnet's recipe dependencies or run "
                "`python -m pip install 'POT>=0.9.7,<0.10'`."
            ) from error

        source_float = source.float()
        target_float = target.float()
        if self.normalize_by_dimension:
            scale = math.sqrt(source.size(1))
            source_float = source_float / scale
            target_float = target_float / scale

        result = ot.solve_sample(
            source_float,
            target_float,
            a=source_mass,
            b=target_mass,
            metric="sqeuclidean",
            reg=self.epsilon,
            reg_type="KL",
            max_iter=self.num_iterations,
            tol=self.stop_threshold,
            grad="envelope",
        )
        return result.value

    def forward(
        self,
        source: torch.Tensor,
        reference: torch.Tensor,
        source_mass: Optional[torch.Tensor] = None,
        reference_mass: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if source.dim() != 2 or reference.dim() != 2:
            raise ValueError(
                "source and reference must both have shape (num_points, dim)"
            )
        if source.size(1) != reference.size(1):
            raise ValueError(
                "source and reference dimensions must match: "
                f"{source.size(1)} != {reference.size(1)}"
            )
        if source.size(0) == 0 or reference.size(0) == 0:
            raise ValueError("source and reference must contain at least one point")

        normalized_source_mass = self._normalize_mass(
            source, source_mass, "source_mass"
        )
        normalized_reference_mass = self._normalize_mass(
            reference, reference_mass, "reference_mass"
        )
        cross_cost = self._regularized_ot(
            source,
            reference,
            normalized_source_mass,
            normalized_reference_mass,
        )
        source_self_cost = self._regularized_ot(
            source,
            source,
            normalized_source_mass,
            normalized_source_mass,
        )

        if reference_mass is None:
            if not torch.isfinite(self.reference_self_cost):
                with torch.no_grad():
                    reference_self_cost = self._regularized_ot(
                        reference,
                        reference,
                        normalized_reference_mass,
                        normalized_reference_mass,
                    )
                    self.reference_self_cost.copy_(reference_self_cost)
            reference_self_cost = self.reference_self_cost
        else:
            # Explicit reference masses may change between calls, so their
            # self-cost must not reuse the uniform-reference cache.
            reference_self_cost = self._regularized_ot(
                reference,
                reference,
                normalized_reference_mass,
                normalized_reference_mass,
            )

        return cross_cost - 0.5 * source_self_cost - 0.5 * reference_self_cost
