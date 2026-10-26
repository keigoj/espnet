from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.layers.sinkhorn import UniformSinkhornDivergence
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class GumbelKMeansOutput(NamedTuple):
    """Outputs from :class:`GumbelKMeans`."""

    quantized: torch.Tensor
    indices: torch.Tensor
    probabilities: torch.Tensor
    distortion_loss: torch.Tensor
    commitment_loss: torch.Tensor
    diversity_loss: torch.Tensor
    codebook_perplexity: torch.Tensor
    temperature: torch.Tensor


class GumbelKMeans(torch.nn.Module):
    """Differentiable K-means using Gumbel-Softmax assignments.

    During training, assignments are sampled with Gumbel-Softmax.  Setting
    ``hard=True`` uses the straight-through estimator: the forward pass is a
    one-hot assignment while the backward pass uses the soft probabilities.
    Deterministic hard assignments also use the straight-through estimator
    during training. Evaluation and explicit unit extraction are deterministic.

    The module keeps the codebook as a regular ``nn.Parameter``.  It is
    therefore synchronized by DDP and included in ESPnet checkpoints.
    Temperature annealing pauses while the codebook is frozen, so staged
    training starts annealing when centroid optimization begins.
    """

    @typechecked
    def __init__(
        self,
        input_dim: int,
        num_clusters: int,
        temperature: float = 2.0,
        min_temperature: float = 0.5,
        temperature_decay: float = 0.999995,
        hard: bool = True,
        normalize_inputs: bool = False,
        logit_scale: float = 1.0,
        initial_centroid_path: Optional[str] = None,
        reference_centroid_path: Optional[str] = None,
        initialize_from_reference: bool = True,
        ot_epsilon: float = 1.0,
        ot_iterations: int = 20,
        ot_mass_mode: str = "uniform",
        eps: float = 1.0e-7,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive: {input_dim}")
        if num_clusters <= 1:
            raise ValueError(f"num_clusters must be greater than one: {num_clusters}")
        if temperature <= 0.0 or min_temperature <= 0.0:
            raise ValueError("Gumbel-Softmax temperatures must be positive")
        if not 0.0 < temperature_decay <= 1.0:
            raise ValueError(
                f"temperature_decay must be in (0, 1]: {temperature_decay}"
            )
        if logit_scale <= 0.0:
            raise ValueError(f"logit_scale must be positive: {logit_scale}")
        if ot_mass_mode != "uniform":
            raise NotImplementedError(
                "Only ot_mass_mode='uniform' is implemented. The option is a "
                "placeholder for future cluster-usage or reference-frequency masses."
            )

        self.input_dim = input_dim
        self.num_clusters = num_clusters
        self.initial_temperature = temperature
        self.min_temperature = min_temperature
        self.temperature_decay = temperature_decay
        self.hard = hard
        self.normalize_inputs = normalize_inputs
        self.logit_scale = logit_scale
        self.initialize_from_reference = initialize_from_reference
        self.ot_mass_mode = ot_mass_mode
        self.eps = eps

        self.codebook = torch.nn.Parameter(torch.empty(num_clusters, input_dim))
        torch.nn.init.normal_(self.codebook, mean=0.0, std=input_dim**-0.5)
        self.register_buffer("initial_codebook", None, persistent=False)
        self.register_buffer("reference_codebook", None)
        self.ot_divergence = UniformSinkhornDivergence(
            epsilon=ot_epsilon,
            num_iterations=ot_iterations,
            normalize_by_dimension=True,
        )
        if initial_centroid_path is not None:
            initial = self._load_centroids(initial_centroid_path, "Initial")
            self._validate_centroid_shape(initial, initial_centroid_path, "Initial")
            self.initial_codebook = initial
        if reference_centroid_path is not None:
            reference = self._load_centroids(reference_centroid_path, "Reference")
            self._validate_centroid_shape(
                reference, reference_centroid_path, "Reference"
            )
            self.reference_codebook = reference

        # An explicit initialization file takes precedence.  The legacy
        # initialize_from_reference path remains unchanged when it is omitted.
        if self.initial_codebook is not None:
            self.set_codebook(self.initial_codebook)
        elif self.initialize_from_reference and self.reference_codebook is not None:
            self.set_codebook(self.reference_codebook)
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))

    @staticmethod
    def _load_centroids(path: str, kind: str) -> torch.Tensor:
        centroid_path = Path(path)
        if not centroid_path.is_file():
            raise FileNotFoundError(f"{kind} centroid file not found: {path}")
        loaded = np.load(centroid_path)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            try:
                if "centers" in loaded.files:
                    array = loaded["centers"]
                elif len(loaded.files) == 1:
                    array = loaded[loaded.files[0]]
                else:
                    raise ValueError(
                        f"{kind} .npz must contain 'centers' or exactly one array: "
                        f"{path}"
                    )
            finally:
                loaded.close()
        else:
            array = loaded
        centroids = torch.as_tensor(np.asarray(array), dtype=torch.float32)
        if centroids.dim() != 2:
            raise ValueError(
                f"{kind} centroids must have shape (num_clusters, input_dim): "
                f"{tuple(centroids.shape)}"
            )
        if not torch.isfinite(centroids).all():
            raise ValueError(f"{kind} centroids contain non-finite values: {path}")
        return centroids

    def _validate_centroid_shape(
        self, centroids: torch.Tensor, path: str, kind: str
    ) -> None:
        if centroids.shape != self.codebook.shape:
            raise ValueError(
                f"{kind} centroids must match the learnable codebook shape: "
                f"expected {tuple(self.codebook.shape)}, got "
                f"{tuple(centroids.shape)} from {path}"
            )

    @property
    def current_temperature(self) -> float:
        temperature = self.initial_temperature * (
            self.temperature_decay ** int(self.num_updates.item())
        )
        return max(self.min_temperature, temperature)

    def set_num_updates(self, num_updates: int) -> None:
        """Set the temperature scheduler step, e.g. when resuming externally."""
        if num_updates < 0:
            raise ValueError(f"num_updates must be non-negative: {num_updates}")
        self.num_updates.fill_(num_updates)

    @torch.no_grad()
    def set_codebook(self, centers: torch.Tensor) -> None:
        """Initialize the learnable centers from an offline K-means model."""
        if centers.shape != self.codebook.shape:
            raise ValueError(
                f"Expected centers with shape {tuple(self.codebook.shape)}, "
                f"but got {tuple(centers.shape)}"
            )
        self.codebook.copy_(centers.to(self.codebook))

    def optimal_transport_loss(
        self,
        source_mass: Optional[torch.Tensor] = None,
        reference_mass: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compare codebooks, optionally using future non-uniform masses.

        Current training calls this method without masses and therefore uses
        ``ot_mass_mode='uniform'``.  The optional arguments are the extension
        point for cluster-usage and reference-frequency weighting.
        """
        if self.reference_codebook is None:
            raise RuntimeError(
                "reference_centroid_path is required to compute optimal transport loss"
            )
        return self.ot_divergence(
            self.codebook,
            self.reference_codebook,
            source_mass=source_mass,
            reference_mass=reference_mass,
        )

    @torch.no_grad()
    def espnet_initialization_fn(self) -> None:
        """Restore centroid initialization after ESPnet's global initializer."""
        if self.initial_codebook is not None:
            self.set_codebook(self.initial_codebook)
        elif self.initialize_from_reference and self.reference_codebook is not None:
            self.set_codebook(self.reference_codebook)

    def _distances(self, inputs: torch.Tensor) -> torch.Tensor:
        # Squared Euclidean distance without materializing (x - c) for every
        # cluster.  Disable autocast as well as casting the inputs: otherwise
        # matmul is downcast again and cancellation can erase nearby distances.
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            inputs_float = inputs.float()
            codebook_float = self.codebook.float()
            if self.normalize_inputs:
                inputs_float = F.normalize(inputs_float, dim=-1)
                codebook_float = F.normalize(codebook_float, dim=-1)
            input_norm = inputs_float.square().sum(dim=-1, keepdim=True)
            center_norm = codebook_float.square().sum(dim=-1)
            distances = (
                input_norm
                + center_norm
                - 2.0 * torch.matmul(inputs_float, codebook_float.transpose(0, 1))
            )
            return distances.clamp_min_(0.0)

    def _valid_mask(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        if input_lengths.dim() != 1:
            raise ValueError(
                f"input_lengths must have one dimension: {input_lengths.shape}"
            )
        if inputs.size(0) != input_lengths.size(0):
            raise ValueError(
                "Batch size mismatch between inputs and input_lengths: "
                f"{inputs.shape}, {input_lengths.shape}"
            )
        return ~make_pad_mask(input_lengths, maxlen=inputs.size(1)).to(inputs.device)

    def forward(
        self,
        inputs: torch.Tensor,
        input_lengths: torch.Tensor,
        sample_gumbel: Optional[bool] = None,
    ) -> GumbelKMeansOutput:
        """Quantize a padded batch of SSL features.

        Args:
            inputs: SSL features with shape ``(batch, time, input_dim)``.
            input_lengths: Valid frame counts with shape ``(batch,)``.
            sample_gumbel: Override stochastic training behavior.  ``False``
                uses nearest-centroid assignments, retaining straight-through
                gradients when training with ``hard=True``.
        """
        if inputs.dim() != 3 or inputs.size(-1) != self.input_dim:
            raise ValueError(
                "Expected inputs shaped (batch, time, "
                f"{self.input_dim}), but got {tuple(inputs.shape)}"
            )

        valid_mask = self._valid_mask(inputs, input_lengths)
        distances = self._distances(inputs)
        # Apply this before Gumbel noise is sampled.  A positive scale keeps
        # deterministic nearest-centroid assignments unchanged while
        # controlling the relative strength of distances and Gumbel noise.
        logits = -distances * self.logit_scale
        temperature_value = self.current_temperature
        use_gumbel = self.training if sample_gumbel is None else sample_gumbel
        soft_probabilities = F.softmax(logits / temperature_value, dim=-1)

        if use_gumbel:
            probabilities = F.gumbel_softmax(
                logits,
                tau=temperature_value,
                hard=self.hard,
                dim=-1,
            )
        elif self.hard:
            indices = logits.argmax(dim=-1)
            probabilities = F.one_hot(indices, self.num_clusters).to(logits.dtype)
            if self.training:
                # Keep nearest-centroid values in the forward pass, but let
                # ASR gradients reach the SSL features without Gumbel noise.
                probabilities = (
                    probabilities - soft_probabilities.detach() + soft_probabilities
                )
        else:
            probabilities = soft_probabilities

        valid_probabilities = probabilities * valid_mask.unsqueeze(-1)
        valid_soft_probabilities = soft_probabilities * valid_mask.unsqueeze(-1)
        quantized = torch.matmul(
            valid_probabilities.to(self.codebook.dtype), self.codebook
        ).to(inputs.dtype)
        indices = probabilities.argmax(dim=-1).masked_fill(~valid_mask, -1)

        valid_count = valid_mask.sum().clamp_min(1).to(distances.dtype)
        loss_normalizer = valid_count * self.input_dim
        # Equation (3) in "Differentiable K-means for Fully-optimized
        # Discrete Token-based ASR": ||s_i - h_tilde_i M||_2^2.  With
        # hard=True, quantized contains the selected centroid in the forward
        # pass while gradients follow the soft Gumbel-Softmax assignments.
        distortion_squared_error = (
            (inputs.float() - quantized.float()).square().sum(dim=-1)
        )
        distortion_loss = (
            distortion_squared_error * valid_mask
        ).sum() / loss_normalizer

        squared_error = (
            (inputs.float() - quantized.detach().float()).square().sum(dim=-1)
        )
        commitment_loss = (squared_error * valid_mask).sum() / loss_normalizer

        average_probabilities = valid_soft_probabilities.sum(dim=(0, 1)) / valid_count
        average_probabilities = average_probabilities.clamp_min(self.eps)
        entropy = -(average_probabilities * average_probabilities.log()).sum()
        codebook_perplexity = entropy.exp()
        diversity_loss = (
            torch.log(
                torch.tensor(
                    float(self.num_clusters),
                    device=entropy.device,
                    dtype=entropy.dtype,
                )
            )
            - entropy
        )

        temperature = torch.tensor(
            temperature_value, device=inputs.device, dtype=torch.float32
        )
        if (
            self.training
            and sample_gumbel is not False
            and self.codebook.requires_grad
        ):
            self.num_updates.add_(1)

        return GumbelKMeansOutput(
            quantized=quantized,
            indices=indices,
            probabilities=valid_probabilities,
            distortion_loss=distortion_loss,
            commitment_loss=commitment_loss,
            diversity_loss=diversity_loss,
            codebook_perplexity=codebook_perplexity,
            temperature=temperature,
        )
