from typing import Dict, List, Optional, Tuple

import torch
from typeguard import typechecked

from espnet2.layers.gumbel_kmeans import GumbelKMeans, GumbelKMeansOutput
from espnet2.ssl.loss.abs_loss import AbsSSLLoss
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class GumbelKMeansLoss(AbsSSLLoss):
    """Jointly optimize an SSL encoder and differentiable K-means."""

    @typechecked
    def __init__(
        self,
        encoder_output_size: int,
        num_clusters: int,
        layer: int = -1,
        kmeans_weight: float = 1.0,
        commitment_weight: float = 0.0,
        diversity_weight: float = 0.1,
        variance_weight: float = 0.0,
        ot_weight: float = 0.0,
        variance_target: float = 1.0,
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
    ):
        super().__init__()
        if (
            min(
                kmeans_weight,
                commitment_weight,
                diversity_weight,
                variance_weight,
                ot_weight,
            )
            < 0.0
        ):
            raise ValueError("Gumbel K-means loss weights must be non-negative")
        if variance_target <= 0.0:
            raise ValueError("variance_target must be positive")
        if ot_weight > 0.0 and reference_centroid_path is None:
            raise ValueError(
                "reference_centroid_path is required when ot_weight is positive"
            )
        self.layer = layer
        self.kmeans_weight = kmeans_weight
        self.commitment_weight = commitment_weight
        self.diversity_weight = diversity_weight
        self.variance_weight = variance_weight
        self.ot_weight = ot_weight
        self.variance_target = variance_target
        self.quantizer = GumbelKMeans(
            input_dim=encoder_output_size,
            num_clusters=num_clusters,
            temperature=temperature,
            min_temperature=min_temperature,
            temperature_decay=temperature_decay,
            hard=hard,
            normalize_inputs=normalize_inputs,
            logit_scale=logit_scale,
            initial_centroid_path=initial_centroid_path,
            reference_centroid_path=reference_centroid_path,
            initialize_from_reference=initialize_from_reference,
            ot_epsilon=ot_epsilon,
            ot_iterations=ot_iterations,
            ot_mass_mode=ot_mass_mode,
        )

        self.util_attributes = []
        self.required_inputs = ["encoder_output", "encoder_output_lengths"]

    def _select_layer(self, encoder_output: List[torch.Tensor]) -> torch.Tensor:
        if not -len(encoder_output) <= self.layer < len(encoder_output):
            raise ValueError(
                f"Requested SSL layer {self.layer}, but only "
                f"{len(encoder_output)} outputs are available"
            )
        return encoder_output[self.layer]

    def _feature_variance_loss(
        self, inputs: torch.Tensor, input_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Penalize collapsed SSL dimensions over the valid batch frames."""
        valid_mask = ~make_pad_mask(input_lengths, maxlen=inputs.size(1)).to(
            inputs.device
        )
        valid_inputs = inputs[valid_mask].float()
        if valid_inputs.size(0) < 2:
            return inputs.float().sum() * 0.0
        feature_std = torch.sqrt(valid_inputs.var(dim=0, unbiased=False) + 1.0e-4)
        return torch.relu(self.variance_target - feature_std).mean()

    def _total_loss(
        self,
        output: GumbelKMeansOutput,
        variance_loss: torch.Tensor,
        ot_loss: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.kmeans_weight * output.distortion_loss
            + self.commitment_weight * output.commitment_loss
            + self.diversity_weight * output.diversity_loss
            + self.variance_weight * variance_loss
            + self.ot_weight * ot_loss
        )

    def _stats(
        self,
        output: GumbelKMeansOutput,
        variance_loss: torch.Tensor,
        ot_loss: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return {
            "gumbel_kmeans_loss": self._total_loss(
                output, variance_loss, ot_loss
            ).detach(),
            "gumbel_kmeans_distortion": output.distortion_loss.detach(),
            "gumbel_kmeans_commitment": output.commitment_loss.detach(),
            "gumbel_kmeans_diversity": output.diversity_loss.detach(),
            "gumbel_kmeans_variance": variance_loss.detach(),
            "gumbel_kmeans_ot": ot_loss.detach(),
            "gumbel_kmeans_perplexity": output.codebook_perplexity.detach(),
            "gumbel_kmeans_temperature": output.temperature.detach(),
        }

    def forward(
        self,
        encoder_output: List[torch.Tensor],
        encoder_output_lengths: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict]:
        if encoder_output_lengths is None:
            raise ValueError("encoder_output_lengths is required")
        selected = self._select_layer(encoder_output)
        output = self.quantizer(selected, encoder_output_lengths)
        variance_loss = self._feature_variance_loss(selected, encoder_output_lengths)
        if self.ot_weight > 0.0:
            ot_loss = self.quantizer.optimal_transport_loss()
        else:
            ot_loss = output.distortion_loss.new_zeros(())
        return self._total_loss(output, variance_loss, ot_loss), self._stats(
            output, variance_loss, ot_loss
        )

    @torch.no_grad()
    def extract_units(
        self,
        final_output: torch.Tensor,
        layer_outputs: List[torch.Tensor],
        output_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract deterministic hard units from an SSL encoder output."""
        if self.layer == -1:
            selected = final_output
        else:
            if not -len(layer_outputs) <= self.layer < len(layer_outputs):
                raise ValueError(
                    f"Requested SSL layer {self.layer}, but only "
                    f"{len(layer_outputs)} outputs are available"
                )
            selected = layer_outputs[self.layer]
        output = self.quantizer(selected, output_lengths, sample_gumbel=False)
        return output.indices, output_lengths
