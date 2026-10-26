import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.gumbel_kmeans_asr_model import ESPnetGumbelKMeansASRModel


def test_gumbel_kmeans_frontend_amp_without_encoder_autocast():
    class ReducedPrecisionFrontend(AbsFrontend):
        def output_size(self):
            return 6

        def forward(self, input, input_lengths):
            return input.to(torch.bfloat16), input_lengths

    encoder = TransformerEncoder(
        input_size=6,
        output_size=4,
        attention_heads=1,
        linear_units=4,
        num_blocks=1,
        input_layer="linear",
    )
    model = ESPnetGumbelKMeansASRModel(
        vocab_size=5,
        token_list=["<blank>", "<unk>", "a", "i", "<sos/eos>"],
        frontend=ReducedPrecisionFrontend(),
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=encoder,
        postencoder=None,
        decoder=None,
        ctc=CTC(odim=5, encoder_output_size=4),
        joint_network=None,
        quantizer_conf={"num_clusters": 3},
        ctc_weight=1.0,
        autocast_frontend=True,
        report_cer=False,
        report_wer=False,
    ).eval()

    with torch.inference_mode():
        encoded, lengths = model.encode(torch.randn(2, 10, 6), torch.tensor([10, 8]))
    assert encoded.dtype == torch.float32
    assert torch.isfinite(encoded).all()
    assert torch.equal(lengths, torch.tensor([10, 8]))


def test_gumbel_kmeans_asr_model_backward_and_extract():
    vocab_size = 5
    input_size = 6
    encoder_output_size = 4
    encoder = TransformerEncoder(
        input_size=input_size,
        output_size=encoder_output_size,
        attention_heads=1,
        linear_units=4,
        num_blocks=2,
        input_layer=None,
    )
    decoder = TransformerDecoder(
        vocab_size, encoder_output_size, linear_units=4, num_blocks=2
    )
    ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size)
    model = ESPnetGumbelKMeansASRModel(
        vocab_size=vocab_size,
        token_list=["<blank>", "<unk>", "a", "i", "<sos/eos>"],
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=encoder,
        postencoder=None,
        decoder=decoder,
        ctc=ctc,
        joint_network=None,
        quantizer_conf={"input_dim": input_size, "num_clusters": 3},
        ctc_weight=0.5,
        report_cer=False,
        report_wer=False,
    )

    speech = torch.randn(2, 10, input_size, requires_grad=True)
    speech_lengths = torch.tensor([10, 8], dtype=torch.long)
    text = torch.randint(2, 4, [2, 4], dtype=torch.long)
    text_lengths = torch.tensor([4, 3], dtype=torch.long)
    loss, stats, _ = model(speech, speech_lengths, text, text_lengths)
    loss.backward()

    assert speech.grad is not None
    assert model.quantizer.codebook.grad is not None
    assert "loss_asr" in stats
    assert "loss_gumbel_kmeans" in stats

    model.eval()
    units, unit_lengths = model.encode_units(speech.detach(), speech_lengths)
    assert units.shape == speech.shape[:2]
    assert torch.equal(unit_lengths, speech_lengths)
    assert units[1, 8:].eq(-1).all()

    encoder_output, encoder_lengths = model.encode(speech.detach(), speech_lengths)
    assert encoder_output.size(0) == speech.size(0)
    assert torch.equal(encoder_lengths, speech_lengths)


def test_gumbel_kmeans_asr_only_loss_updates_codebook():
    """ASR supervision must cross the hard straight-through bottleneck."""
    vocab_size = 5
    input_size = 6
    encoder_output_size = 4
    encoder = TransformerEncoder(
        input_size=input_size,
        output_size=encoder_output_size,
        attention_heads=1,
        linear_units=4,
        num_blocks=2,
        input_layer=None,
    )
    decoder = TransformerDecoder(
        vocab_size, encoder_output_size, linear_units=4, num_blocks=2
    )
    ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size)
    model = ESPnetGumbelKMeansASRModel(
        vocab_size=vocab_size,
        token_list=["<blank>", "<unk>", "a", "i", "<sos/eos>"],
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=encoder,
        postencoder=None,
        decoder=decoder,
        ctc=ctc,
        joint_network=None,
        quantizer_conf={
            "input_dim": input_size,
            "num_clusters": 3,
            "hard": True,
        },
        asr_weight=1.0,
        kmeans_weight=0.0,
        commitment_weight=0.0,
        diversity_weight=0.0,
        ot_weight=0.0,
        ctc_weight=0.5,
        report_cer=False,
        report_wer=False,
    )

    speech = torch.randn(2, 10, input_size, requires_grad=True)
    speech_lengths = torch.tensor([10, 8], dtype=torch.long)
    text = torch.randint(2, 4, [2, 4], dtype=torch.long)
    text_lengths = torch.tensor([4, 3], dtype=torch.long)
    loss, stats, _ = model(speech, speech_lengths, text, text_lengths)
    loss.backward()

    assert stats["loss_gumbel_kmeans"].item() == 0.0
    assert model.quantizer.codebook.grad is not None
    assert model.quantizer.codebook.grad.abs().sum().item() > 0.0
    assert speech.grad is not None
    assert speech.grad.abs().sum().item() > 0.0


def test_gumbel_kmeans_asr_model_can_force_deterministic_training_assignments():
    vocab_size = 5
    input_size = 6
    encoder_output_size = 4
    encoder = TransformerEncoder(
        input_size=input_size,
        output_size=encoder_output_size,
        attention_heads=1,
        linear_units=4,
        num_blocks=2,
        input_layer=None,
    )
    decoder = TransformerDecoder(
        vocab_size, encoder_output_size, linear_units=4, num_blocks=2
    )
    ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size)
    model = ESPnetGumbelKMeansASRModel(
        vocab_size=vocab_size,
        token_list=["<blank>", "<unk>", "a", "i", "<sos/eos>"],
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=encoder,
        postencoder=None,
        decoder=decoder,
        ctc=ctc,
        joint_network=None,
        quantizer_conf={"input_dim": input_size, "num_clusters": 3},
        quantizer_sample_gumbel=False,
        kmeans_weight=0.0,
        diversity_weight=0.0,
        ctc_weight=0.5,
        report_cer=False,
        report_wer=False,
    )
    model.train()
    speech = torch.randn(2, 10, input_size, requires_grad=True)
    speech_lengths = torch.tensor([10, 8], dtype=torch.long)

    first, _ = model._extract_feats(speech, speech_lengths)
    second, _ = model._extract_feats(speech, speech_lengths)

    assert torch.equal(first, second)
    assert model.quantizer.num_updates.item() == 0

    text = torch.randint(2, 4, [2, 4], dtype=torch.long)
    loss, stats, _ = model(speech, speech_lengths, text, torch.tensor([4, 3]))
    loss.backward()
    assert stats["loss_gumbel_kmeans"].item() == 0.0
    assert speech.grad is not None
    assert torch.isfinite(speech.grad).all()
    assert speech.grad.abs().sum() > 0
    assert model.quantizer.codebook.grad is not None
    assert torch.isfinite(model.quantizer.codebook.grad).all()
    assert model.quantizer.codebook.grad.abs().sum() > 0
