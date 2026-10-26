import torch

from espnet2.asr.encoder.identity_encoder import IdentityEncoder


def test_identity_encoder_preserves_features_and_lengths():
    encoder = IdentityEncoder(input_size=4)
    inputs = torch.randn(2, 5, 4)
    lengths = torch.tensor([5, 3], dtype=torch.long)

    (output, layer_outputs), output_lengths, _ = encoder(
        inputs, lengths, return_all_hs=True
    )

    assert output is inputs
    assert layer_outputs == [inputs]
    assert torch.equal(output_lengths, lengths)
    assert encoder.output_size() == 4
