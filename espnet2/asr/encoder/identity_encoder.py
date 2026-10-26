from typing import Optional, Tuple, Union

import torch
from typeguard import typechecked

from espnet2.asr.ctc import CTC
from espnet2.asr.encoder.abs_encoder import AbsEncoder


class IdentityEncoder(AbsEncoder):
    """Pass frontend features through without changing their representation."""

    @typechecked
    def __init__(self, input_size: int):
        super().__init__()
        self._output_size = input_size

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        ctc: Optional[CTC] = None,
        return_all_hs: bool = False,
    ) -> Tuple[
        Union[torch.Tensor, Tuple[torch.Tensor, list]],
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        del masks, ctc
        if return_all_hs:
            # ESPnetSSLModel expects the same (final, layer_outputs) structure
            # returned by TransformerEncoder when all hidden states are requested.
            return (xs_pad, [xs_pad]), ilens, prev_states
        return xs_pad, ilens, prev_states
