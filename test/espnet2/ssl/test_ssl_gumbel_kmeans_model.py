import torch

from espnet2.asr.encoder.identity_encoder import IdentityEncoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.frontend.cnn import CNNFrontend
from espnet2.asr.preencoder.linear import LinearProjection
from espnet2.ssl.espnet_model import ESPnetSSLModel
from espnet2.ssl.loss.gumbel_kmeans import GumbelKMeansLoss
from espnet2.ssl.loss.hubert import HuBERTLoss
from espnet2.ssl.utils.mask import Masking


def _build_ssl_model(losses, util_modules=None):
    util_modules = util_modules or torch.nn.ModuleDict()
    util_attributes = set(util_modules.keys())
    required_inputs = set()
    for loss_function in losses:
        required_inputs.update(loss_function.required_inputs)

    frontend = CNNFrontend("group_norm", "standard", True, [(3, 3, 2)])
    preencoder = LinearProjection(3, 16)
    encoder = TransformerEncoder(
        16,
        output_size=16,
        attention_heads=1,
        linear_units=4,
        num_blocks=2,
        pos_enc_layer_type="conv",
        input_layer="wav2vec",
    )
    return ESPnetSSLModel(
        frontend=frontend,
        specaug=None,
        normalize=None,
        preencoder=preencoder,
        encoder=encoder,
        losses=losses,
        util_attributes=util_attributes,
        required_inputs=required_inputs,
        util_modules=util_modules,
        token_list=["0", "1", "2"],
    )


def test_ssl_and_gumbel_kmeans_train_together():
    gumbel_loss = GumbelKMeansLoss(
        encoder_output_size=16,
        num_clusters=3,
        diversity_weight=0.1,
    )
    model = _build_ssl_model([gumbel_loss])
    speech = torch.randn(2, 32, requires_grad=True)
    speech_lengths = torch.tensor([32, 24], dtype=torch.long)

    loss, stats, _ = model(speech, speech_lengths)
    loss.backward()

    assert speech.grad is not None
    assert next(model.encoder.parameters()).grad is not None
    assert gumbel_loss.quantizer.codebook.grad is not None
    assert "gumbel_kmeans_loss" in stats

    model.eval()
    units, unit_lengths = model.encode_units(speech.detach(), speech_lengths)
    assert units.size(0) == 2
    assert torch.equal(unit_lengths, torch.tensor([15, 11]))


def test_hubert_and_gumbel_kmeans_losses_train_together():
    hubert_loss = HuBERTLoss(
        encoder_output_size=16,
        num_classes=5,
        final_dim=4,
    )
    gumbel_loss = GumbelKMeansLoss(
        encoder_output_size=16,
        num_clusters=3,
    )
    model = _build_ssl_model(
        [hubert_loss, gumbel_loss],
        torch.nn.ModuleDict({"mask": Masking(encoder_embed_dim=16)}),
    )
    speech = torch.randn(2, 32, requires_grad=True)
    speech_lengths = torch.tensor([32, 24], dtype=torch.long)
    text = torch.randint(0, 5, (2, 15), dtype=torch.long)
    text[1, 11:] = -1
    text_lengths = torch.tensor([15, 11], dtype=torch.long)

    loss, stats, _ = model(speech, speech_lengths, text, text_lengths)
    loss.backward()

    assert speech.grad is not None
    assert next(model.encoder.parameters()).grad is not None
    assert gumbel_loss.quantizer.codebook.grad is not None
    assert any(key.startswith("hubert_loss_m_") for key in stats)
    assert "gumbel_kmeans_loss" in stats


def test_identity_encoder_quantizes_frontend_space_directly():
    gumbel_loss = GumbelKMeansLoss(
        encoder_output_size=4,
        num_clusters=3,
        diversity_weight=0.1,
    )
    model = ESPnetSSLModel(
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=IdentityEncoder(input_size=4),
        losses=[gumbel_loss],
        util_attributes=set(),
        required_inputs=set(gumbel_loss.required_inputs),
        util_modules=torch.nn.ModuleDict(),
        token_list=["0", "1", "2"],
    )
    features = torch.randn(2, 5, 4, requires_grad=True)
    lengths = torch.tensor([5, 3], dtype=torch.long)

    loss, _, _ = model(features, lengths)
    loss.backward()

    assert features.grad is not None
    assert gumbel_loss.quantizer.codebook.grad is not None
    model.eval()
    units, unit_lengths = model.encode_units(features.detach(), lengths)
    assert units.shape == features.shape[:2]
    assert torch.equal(unit_lengths, lengths)
