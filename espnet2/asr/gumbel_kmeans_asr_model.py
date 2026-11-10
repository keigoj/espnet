from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel, autocast, autocast_type
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.gumbel_kmeans import GumbelKMeans, GumbelKMeansOutput
from espnet2.torch_utils.device_funcs import force_gatherable


class ESPnetGumbelKMeansASRModel(ESPnetASRModel):
    """ASR model with differentiable K-means after an SSL frontend.

    A typical data flow is ``waveform -> SSL frontend -> Gumbel K-means ->
    ASR encoder -> decoder``.  ASR and cluster losses both update the SSL
    frontend unless it is explicitly frozen with ESPnet's ``freeze_param``.
    Optional severity-invariant regularization (SIR) matches utterance-level
    soft token distributions to content-parallel reference speech, with gradients
    stopped on the reference side (Wang et al., arXiv:2609.16855, Eq. 8).
    References can be healthy speech or, as an experimental extension, mild
    dysarthric speech. Unpaired utterances can be excluded from SIR only.
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: Optional[AbsEncoder],
        postencoder: Optional[AbsPostEncoder],
        decoder: Optional[AbsDecoder],
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        quantizer_conf: Optional[dict] = None,
        asr_weight: float = 1.0,
        kmeans_weight: float = 1.0,
        commitment_weight: float = 0.0,
        diversity_weight: float = 0.1,
        ot_weight: float = 0.0,
        quantizer_sample_gumbel: Optional[bool] = None,
        sir_weight: float = 0.0,
        **kwargs,
    ):
        if (
            min(
                asr_weight,
                kmeans_weight,
                commitment_weight,
                diversity_weight,
                ot_weight,
                sir_weight,
            )
            < 0.0
        ):
            raise ValueError(
                "ASR, Gumbel K-means, and SIR weights must be non-negative"
            )
        if (
            asr_weight
            + kmeans_weight
            + commitment_weight
            + diversity_weight
            + ot_weight
            + sir_weight
            == 0.0
        ):
            raise ValueError("At least one loss weight must be positive")

        super().__init__(
            vocab_size=vocab_size,
            token_list=token_list,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=joint_network,
            **kwargs,
        )

        quantizer_conf = dict(quantizer_conf or {})
        if "input_dim" not in quantizer_conf:
            if frontend is None:
                raise ValueError(
                    "quantizer_conf.input_dim is required when frontend is null"
                )
            quantizer_conf["input_dim"] = frontend.output_size()
        self.quantizer = GumbelKMeans(**quantizer_conf)
        self.asr_weight = asr_weight
        self.kmeans_weight = kmeans_weight
        self.commitment_weight = commitment_weight
        self.diversity_weight = diversity_weight
        self.ot_weight = ot_weight
        self.sir_weight = sir_weight
        # None follows the quantizer's normal behavior: stochastic in training
        # and deterministic in evaluation.  False is used by a truly fixed
        # nearest-centroid baseline while the downstream ASR remains in train mode.
        self.quantizer_sample_gumbel = quantizer_sample_gumbel
        if self.ot_weight > 0.0 and self.quantizer.reference_codebook is None:
            raise ValueError(
                "quantizer_conf.reference_centroid_path is required when "
                "ot_weight is positive"
            )
        self._last_quantizer_output: Optional[GumbelKMeansOutput] = None
        self._capture_quantizer_output = False
        self._capture_sir_output = False

    def _extract_frontend_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run only the regular ESPnet frontend, before quantization."""
        return super()._extract_feats(speech, speech_lengths)

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feats, feats_lengths = self._extract_frontend_feats(speech, speech_lengths)
        output = self.quantizer(
            feats,
            feats_lengths,
            sample_gumbel=self.quantizer_sample_gumbel,
            return_soft_probabilities=self._capture_sir_output,
        )
        if self._capture_quantizer_output:
            self._last_quantizer_output = output
        return output.quantized, feats_lengths

    def _cluster_loss(
        self, output: GumbelKMeansOutput, ot_loss: torch.Tensor
    ) -> torch.Tensor:
        return (
            self.kmeans_weight * output.distortion_loss
            + self.commitment_weight * output.commitment_loss
            + self.diversity_weight * output.diversity_loss
            + self.ot_weight * ot_loss
        )

    def _utterance_token_distribution(
        self,
        output: GumbelKMeansOutput,
        pair_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Average relaxed assignments over valid frames, in float32."""
        if output.soft_probabilities is None:
            raise RuntimeError("SIR requires relaxed Gumbel-Softmax probabilities")
        probabilities = output.soft_probabilities
        if pair_mask is not None:
            probabilities = probabilities[pair_mask]
        probabilities = probabilities.float().sum(dim=1)
        mass = probabilities.sum(dim=-1, keepdim=True)
        if (mass <= 0.0).any():
            raise ValueError("SIR requires at least one valid frame per utterance")
        probabilities = (probabilities / mass).clamp_min(self.quantizer.eps)
        return probabilities / probabilities.sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def _healthy_token_distribution(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> torch.Tensor:
        """Compute the stop-gradient target using the same tokenizer."""
        with autocast(self.autocast_frontend, dtype=autocast_type):
            feats, feats_lengths = self._extract_frontend_feats(speech, speech_lengths)
            output = self.quantizer(
                feats,
                feats_lengths,
                sample_gumbel=self.quantizer_sample_gumbel,
                update_temperature=False,
                return_soft_probabilities=True,
            )
        return self._utterance_token_distribution(output)

    def _normalize_sir_loss(
        self, loss: torch.Tensor, num_pairs: int, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Normalize training pairs globally; evaluation stays process-local."""
        counts = loss.new_tensor([num_pairs, batch_size])
        if (
            self.training
            and torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            # Evaluation also runs on rank zero alone for attention plotting.
            # A collective there would wait for workers that never enter it.
            # Every worker participates, including those with zero local pairs.
            torch.distributed.all_reduce(counts)
            # Trainer averages returned losses using primary batch sizes, then
            # compensates for DDP's gradient averaging. Convert the local pair
            # mean to the corresponding contribution to the global pair mean.
            loss = (
                loss * (num_pairs * counts[1]) / (batch_size * counts[0].clamp_min(1))
            )
        return loss, counts[0]

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        healthy_speech: Optional[torch.Tensor] = None,
        healthy_speech_lengths: Optional[torch.Tensor] = None,
        sir_pair_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Compute ASR, clustering, and optional content-parallel SIR losses.

        ``healthy_speech[b]`` must have the same linguistic content as
        ``speech[b]``; frame alignment and equal durations are not required.
        The reference may also be mild dysarthric speech. ``sir_pair_mask``
        has shape ``(batch,)`` or ``(batch, 1)`` and contains 1 for a valid
        pair and 0 otherwise. Omission selects all utterances. ASR and cluster
        losses always use the full batch; SIR averages only over selected pairs.
        An all-zero mask permits training without references. Evaluation may
        omit pairs, in which case SIR is reported as zero.
        """
        healthy_distribution = None
        pair_mask = None
        if self.sir_weight > 0.0:
            if sir_pair_mask is None:
                pair_mask = torch.ones(
                    speech.size(0), dtype=torch.bool, device=speech.device
                )
            else:
                pair_mask = sir_pair_mask
                if pair_mask.dim() == 2 and pair_mask.size(1) == 1:
                    pair_mask = pair_mask.squeeze(1)
                if pair_mask.dim() != 1 or pair_mask.size(0) != speech.size(0):
                    raise ValueError(
                        "sir_pair_mask must have shape (batch,) or (batch, 1)"
                    )
                if pair_mask.device != speech.device:
                    raise ValueError(
                        "sir_pair_mask and speech must be on the same device"
                    )
                if not ((pair_mask == 0) | (pair_mask == 1)).all():
                    raise ValueError("sir_pair_mask must contain only 0 or 1")
                pair_mask = pair_mask.bool()
            if (healthy_speech is None) != (healthy_speech_lengths is None):
                raise ValueError(
                    "healthy_speech and healthy_speech_lengths must be provided together"
                )
            if healthy_speech is None:
                if pair_mask.any() and (self.training or sir_pair_mask is not None):
                    raise ValueError(
                        "Content-parallel healthy_speech and healthy_speech_lengths "
                        "are required for selected SIR pairs"
                    )
            else:
                if (
                    healthy_speech.dim() != speech.dim()
                    or healthy_speech.size(0) != speech.size(0)
                    or healthy_speech.shape[2:] != speech.shape[2:]
                ):
                    raise ValueError(
                        "healthy_speech must match speech batch and feature dimensions"
                    )
                if healthy_speech_lengths.dim() != 1 or healthy_speech_lengths.size(
                    0
                ) != speech.size(0):
                    raise ValueError("healthy_speech_lengths must have shape (batch,)")
                if healthy_speech.device != speech.device:
                    raise ValueError(
                        "healthy_speech and speech must be on the same device"
                    )
                reference_lengths = healthy_speech_lengths[
                    pair_mask.to(healthy_speech_lengths.device)
                ]
                if (reference_lengths <= 0).any() or (
                    reference_lengths > healthy_speech.size(1)
                ).any():
                    raise ValueError(
                        "healthy_speech_lengths must contain a valid positive "
                        "length for each selected SIR pair"
                    )
                if pair_mask.any():
                    # Only selected references enter the frontend. Run first so
                    # both branches use the same temperature; only the main ASR
                    # forward advances the quantizer's annealing schedule.
                    healthy_distribution = self._healthy_token_distribution(
                        healthy_speech[pair_mask], reference_lengths
                    )

        self._last_quantizer_output = None
        self._capture_quantizer_output = True
        self._capture_sir_output = healthy_distribution is not None
        try:
            asr_loss, stats, weight = super().forward(
                speech=speech,
                speech_lengths=speech_lengths,
                text=text,
                text_lengths=text_lengths,
                **kwargs,
            )
        finally:
            self._capture_quantizer_output = False
            self._capture_sir_output = False
        output = self._last_quantizer_output
        self._last_quantizer_output = None
        if output is None:
            raise RuntimeError("Gumbel K-means was not invoked during ASR forward")

        if self.ot_weight > 0.0:
            ot_loss = self.quantizer.optimal_transport_loss()
        else:
            ot_loss = output.distortion_loss.new_zeros(())
        cluster_loss = self._cluster_loss(output, ot_loss)
        sir_loss = output.distortion_loss.new_zeros(())
        local_num_pairs = 0
        if healthy_distribution is not None:
            distribution = self._utterance_token_distribution(output, pair_mask)
            local_num_pairs = healthy_distribution.size(0)
            # F.kl_div(log p_d, p_c) computes KL(p_c || p_d), Eq. (8).
            sir_loss = F.kl_div(
                distribution.log(), healthy_distribution.detach(), reduction="batchmean"
            )
        if self.sir_weight > 0.0:
            sir_loss, sir_num_pairs = self._normalize_sir_loss(
                sir_loss, local_num_pairs, speech.size(0)
            )
        else:
            sir_num_pairs = output.distortion_loss.new_zeros(())
        loss = self.asr_weight * asr_loss + cluster_loss + self.sir_weight * sir_loss
        stats.update(
            {
                "loss_asr": asr_loss.detach(),
                "loss_gumbel_kmeans": cluster_loss.detach(),
                "loss_sir": sir_loss.detach(),
                "sir_num_pairs": sir_num_pairs,
                "gumbel_kmeans_distortion": output.distortion_loss.detach(),
                "gumbel_kmeans_commitment": output.commitment_loss.detach(),
                "gumbel_kmeans_diversity": output.diversity_loss.detach(),
                "gumbel_kmeans_ot": ot_loss.detach(),
                "gumbel_kmeans_perplexity": output.codebook_perplexity.detach(),
                "gumbel_kmeans_temperature": output.temperature.detach(),
                "loss": loss.detach(),
            }
        )
        return force_gatherable((loss, stats, weight), loss.device)

    @torch.no_grad()
    def encode_units(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract deterministic cluster IDs before the downstream ASR encoder."""
        feats, feats_lengths = self._extract_frontend_feats(speech, speech_lengths)
        output = self.quantizer(feats, feats_lengths, sample_gumbel=False)
        return output.indices, feats_lengths
