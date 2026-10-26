from typing import Dict, List, Optional, Tuple, Union

import torch
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
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
        **kwargs,
    ):
        if (
            min(
                asr_weight,
                kmeans_weight,
                commitment_weight,
                diversity_weight,
                ot_weight,
            )
            < 0.0
        ):
            raise ValueError("ASR and Gumbel K-means loss weights must be non-negative")
        if (
            asr_weight
            + kmeans_weight
            + commitment_weight
            + diversity_weight
            + ot_weight
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

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        self._last_quantizer_output = None
        self._capture_quantizer_output = True
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
        output = self._last_quantizer_output
        self._last_quantizer_output = None
        if output is None:
            raise RuntimeError("Gumbel K-means was not invoked during ASR forward")

        if self.ot_weight > 0.0:
            ot_loss = self.quantizer.optimal_transport_loss()
        else:
            ot_loss = output.distortion_loss.new_zeros(())
        cluster_loss = self._cluster_loss(output, ot_loss)
        loss = self.asr_weight * asr_loss + cluster_loss
        stats.update(
            {
                "loss_asr": asr_loss.detach(),
                "loss_gumbel_kmeans": cluster_loss.detach(),
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
