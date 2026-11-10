"""Embed differentiable token assignments independently of cluster centroids."""

from typing import Tuple

import torch
from typeguard import typechecked

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding


class TokenEmbedding(AbsFrontend):
    """Support integer IDs and hard straight-through or soft assignments.

    This follows ESPnet's speech-tokenizer ``TokenEmbedding`` at commit
    0c7942d1e0db7828f8adb2154561c28542ae7806, using this checkout's imports.
    B is the utterance count, T the frame count, K the cluster count, and
    E the embedding dimension. Padding distributions must be zero.
    """

    @typechecked
    def __init__(
        self,
        input_size: int,
        embed_dim: int = 512,
        use_positional_encoding: bool = True,
        positional_dropout_rate: float = 0.1,
    ):
        super().__init__()
        if input_size <= 0 or embed_dim <= 0:
            raise ValueError("input_size and embed_dim must be positive")
        if not 0.0 <= positional_dropout_rate <= 1.0:
            raise ValueError("positional_dropout_rate must be in [0, 1]")
        self.input_size = input_size
        self.embed_dim = embed_dim
        self.embed = torch.nn.Embedding(input_size, embed_dim)
        self.pos_enc = (
            PositionalEncoding(embed_dim, positional_dropout_rate)
            if use_positional_encoding
            else torch.nn.Identity()
        )

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Map IDs (B, T) or assignments (B, T, K) to features (B, T, E).

        Assignment probabilities are normalized along K by the quantizer;
        all-zero padded frames are excluded downstream using lengths (B,).
        """
        if input_lengths.dim() != 1 or input.size(0) != input_lengths.size(0):
            raise ValueError("input_lengths must have shape (B,)")
        if torch.is_floating_point(input):
            if input.dim() != 3 or input.size(-1) != self.input_size:
                raise ValueError("Floating-point assignments must have shape (B, T, K)")
            # (B, T, K) @ (K, E) -> (B, T, E); retain assignment gradients.
            x = torch.matmul(input.to(self.embed.weight.dtype), self.embed.weight)
        else:
            if input.dim() != 2 or input.dtype == torch.bool:
                raise ValueError("Integer token IDs must have shape (B, T)")
            x = self.embed(input.long())  # (B, T) -> (B, T, E).
        return self.pos_enc(x), input_lengths

    def output_size(self) -> int:
        return self.embed_dim
