import copy
from types import SimpleNamespace

import pytest
import torch

from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.transformer_decoder import TransformerDecoder
from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.gumbel_kmeans_asr_model import (
    ESPnetGumbelKMeansASRModel,
    ESPnetTokenizedASRModel,
)
from espnet2.main_funcs.calculate_all_attentions import calculate_all_attentions
from espnet2.train.trainer import Trainer


class TrainableFrontend(AbsFrontend):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(6, 6)

    def output_size(self):
        return 6

    def forward(self, input, input_lengths):
        return self.projection(input), input_lengths


def test_upstream_embedding_and_sampling_follow_staged_freeze(monkeypatch):
    """Warm-up learns E=4 embeddings; epoch 31 also trains the D=6 tokenizer."""
    torch.manual_seed(0)
    model = ESPnetTokenizedASRModel(
        vocab_size=5,
        token_list=["<blank>", "<unk>", "a", "i", "<sos/eos>"],
        frontend=TrainableFrontend(),
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=TransformerEncoder(
            input_size=4,
            output_size=4,
            attention_heads=1,
            linear_units=4,
            num_blocks=1,
            input_layer="linear",
        ),
        postencoder=None,
        decoder=None,
        ctc=CTC(odim=5, encoder_output_size=4),
        joint_network=None,
        quantizer_conf={"num_clusters": 3, "temperature_decay": 0.9},
        asr_embedding_conf={"embed_dim": 4, "use_positional_encoding": False},
        kmeans_weight=0.0,
        diversity_weight=0.0,
        ctc_weight=1.0,
        report_cer=False,
        report_wer=False,
    ).train()
    calls = []
    original = torch.nn.functional.gumbel_softmax

    def sample(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "gumbel_softmax", sample)
    options = SimpleNamespace(
        staged_freeze_param=["frontend", "quantizer"], staged_freeze_until_epoch=30
    )
    speech = torch.randn(2, 10, 6, requires_grad=True)  # (B=2, T=10, D=6).
    lengths = torch.tensor([10, 8])  # (B,); last two frames are padding.
    text = torch.tensor([[2, 3], [3, 2]])  # (B, L=2 target tokens).
    Trainer._set_staged_freeze_state(model, 30, options)
    features, feature_lengths = model._extract_feats(speech, lengths)
    assert features.shape == (2, 10, 4)  # (B, T, E).
    assert features[1, 8:].eq(0).all()
    assert features.device == speech.device
    assert torch.equal(feature_lengths, lengths)
    repeated, _ = model._extract_feats(speech, lengths)
    torch.testing.assert_close(features, repeated, rtol=0, atol=0)
    loss, _, _ = model(speech, lengths, text, torch.tensor([2, 2]))
    loss.backward()
    assert not calls
    assert model.quantizer.num_updates.item() == 0
    assert model.frontend.projection.weight.grad is None
    assert model.quantizer.codebook.grad is None
    assert speech.grad is None
    assert model.asr_embedding.embed.weight.grad.abs().sum() > 0

    model.zero_grad()
    Trainer._set_staged_freeze_state(model, 31, options)
    loss, _, _ = model(speech, lengths, text, torch.tensor([2, 2]))
    loss.backward()
    assert len(calls) == 1
    assert model.quantizer.num_updates.item() == 1
    for parameter in (
        model.frontend.projection.weight,
        model.quantizer.codebook,
        model.asr_embedding.embed.weight,
    ):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0

    # Evaluation switches back to deterministic assignment without annealing.
    model.eval()
    with torch.no_grad():
        model._extract_feats(speech, lengths)
    assert len(calls) == 1
    assert model.quantizer.num_updates.item() == 1


def _build_sir_model(model_class=ESPnetGumbelKMeansASRModel, **kwargs):
    return model_class(
        vocab_size=5,
        token_list=["<blank>", "<unk>", "a", "i", "<sos/eos>"],
        frontend=TrainableFrontend(),
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=TransformerEncoder(
            input_size=6,
            output_size=4,
            attention_heads=1,
            linear_units=4,
            num_blocks=1,
            input_layer="linear",
        ),
        postencoder=None,
        decoder=None,
        ctc=CTC(odim=5, encoder_output_size=4),
        joint_network=None,
        quantizer_conf={"num_clusters": 3, "temperature_decay": 0.9},
        kmeans_weight=0.0,
        diversity_weight=0.0,
        ctc_weight=1.0,
        report_cer=False,
        report_wer=False,
        **kwargs,
    )


def test_legacy_centroid_checkpoint_still_loads_strictly():
    legacy = _build_sir_model()
    restored = _build_sir_model()
    restored.load_state_dict(legacy.state_dict(), strict=True)
    assert not hasattr(restored, "asr_embedding")
    assert not any(key.startswith("asr_embedding.") for key in legacy.state_dict())


def test_sir_with_independent_embedding_keeps_reference_stop_gradient():
    """SIR uses K-way assignments, independently of the ASR readout."""
    model = _build_sir_model(
        sir_weight=0.2,
        asr_embedding_conf={"embed_dim": 6, "positional_dropout_rate": 0.0},
        model_class=ESPnetTokenizedASRModel,
    )
    speech = torch.randn(2, 10, 6, requires_grad=True)  # (B=2, T=10, D=6).
    reference = torch.randn(2, 7, 6, requires_grad=True)  # (B, T_reference=7, D).
    loss, stats, _ = model(
        speech,
        torch.tensor([10, 8]),
        torch.tensor([[2, 3], [3, 2]]),
        torch.tensor([2, 2]),
        healthy_speech=reference,
        healthy_speech_lengths=torch.tensor([7, 5]),
    )
    loss.backward()
    assert torch.isfinite(stats["loss_sir"]).all()
    assert reference.grad is None
    assert speech.grad is not None and torch.isfinite(speech.grad).all()
    assert model.asr_embedding.embed.weight.grad.abs().sum() > 0
    assert torch.isfinite(model.quantizer.codebook.grad).all()


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is not available"
            ),
        ),
    ],
)
@pytest.mark.parametrize("sir_only", [False, True])
@pytest.mark.parametrize("sample_gumbel", [True, False])
def test_sir_matches_paper_kl_and_updates_only_dysarthric_branch(
    device, sir_only, sample_gumbel
):
    torch.manual_seed(3)
    model = _build_sir_model(
        sir_weight=0.4,
        asr_weight=0.0 if sir_only else 1.0,
        quantizer_sample_gumbel=sample_gumbel,
    ).to(device)
    speech = torch.randn(2, 10, 6, device=device, requires_grad=True)
    healthy = torch.randn(2, 12, 6, device=device, requires_grad=True)
    speech_lengths = torch.tensor([10, 8], device=device)
    healthy_lengths = torch.tensor([7, 12], device=device)
    text = torch.tensor([[2, 3], [3, 2]], device=device)
    text_lengths = torch.tensor([2, 2], device=device)
    outputs = []
    hook = model.quantizer.register_forward_hook(
        lambda module, args, output: outputs.append(output)
    )
    loss, stats, weight = model(
        speech, speech_lengths, text, text_lengths, healthy, healthy_lengths
    )
    hook.remove()

    # Average each utterance separately, excluding its own padding; neither
    # frame alignment nor equal lengths are assumed by Eq. (8).
    def average(output, lengths):
        return torch.stack(
            [
                output.soft_probabilities[b, :n].mean(dim=0)
                for b, n in enumerate(lengths.tolist())
            ]
        )

    pc = average(outputs[0], healthy_lengths).detach()
    pd = average(outputs[1], speech_lengths)
    expected = (pc * (pc.log() - pd.log())).sum(dim=-1).mean()
    torch.testing.assert_close(stats["loss_sir"].squeeze(), expected.detach())
    torch.testing.assert_close(
        loss, model.asr_weight * stats["loss_asr"] + 0.4 * expected
    )
    assert not outputs[0].soft_probabilities.requires_grad
    assert outputs[1].soft_probabilities.requires_grad
    assert outputs[0].temperature == outputs[1].temperature
    assert model.quantizer.num_updates.item() == int(sample_gumbel)
    assert loss.device.type == device
    assert stats["loss_sir"].device == loss.device
    assert weight.item() == 2

    loss.backward()
    assert healthy.grad is None
    for gradient in (
        speech.grad,
        model.frontend.projection.weight.grad,
        model.quantizer.codebook.grad,
    ):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    if not sir_only:
        assert next(model.encoder.parameters()).grad.abs().sum() > 0


def test_sir_is_finite_with_extreme_distances_and_cpu_autocast():
    model = _build_sir_model(sir_weight=0.1, quantizer_sample_gumbel=False)
    speech = torch.full((2, 10, 6), 10000.0, requires_grad=True)
    healthy = -speech.detach()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss, stats, _ = model(
            speech,
            torch.tensor([10, 8]),
            torch.tensor([[2, 3], [3, 2]]),
            torch.tensor([2, 2]),
            healthy,
            torch.tensor([8, 10]),
        )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(stats["loss_sir"])
    assert torch.isfinite(speech.grad).all()
    assert torch.isfinite(model.quantizer.codebook.grad).all()


def test_sir_validation_without_pairs_and_disabled_sir_are_backward_compatible():
    model = _build_sir_model(sir_weight=0.2).eval()
    args = (
        torch.randn(2, 10, 6),
        torch.tensor([10, 8]),
        torch.tensor([[2, 3], [3, 2]]),
        torch.tensor([2, 2]),
    )
    with torch.no_grad():
        loss, stats, _ = model(*args)
        assert stats["loss_sir"].item() == 0.0
        torch.testing.assert_close(loss, stats["loss_asr"])
        units, lengths = model.encode_units(*args[:2])
    assert units.shape == (2, 10)
    assert units[1, 8:].eq(-1).all()
    assert torch.equal(lengths, args[1])

    disabled = _build_sir_model()
    assert disabled.sir_weight == 0.0
    loss, stats, _ = disabled(*args)
    assert stats["loss_sir"].item() == 0.0
    torch.testing.assert_close(loss, stats["loss_asr"])
    assert not any("sir" in key for key in disabled.state_dict())
    model.load_state_dict(disabled.state_dict(), strict=True)


@pytest.mark.parametrize("sir_only", [False, True])
@pytest.mark.parametrize("column_mask", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is not available"
            ),
        ),
    ],
)
def test_sir_partial_pairs_keep_full_asr_batch_and_average_only_selected_pairs(
    sir_only, column_mask, device
):
    torch.manual_seed(3)
    model = _build_sir_model(
        sir_weight=0.4,
        asr_weight=0.0 if sir_only else 1.0,
        quantizer_sample_gumbel=False,
    ).to(device)
    speech = torch.randn(3, 10, 6, device=device, requires_grad=True)
    reference = torch.randn(3, 12, 6, device=device)
    # A missing reference need not even contain usable samples. That row must
    # never enter the frontend or contribute to the SIR loss/normalization.
    reference[1] = float("nan")
    reference.requires_grad_()
    pair_mask = torch.tensor([1, 0, 1], device=device)
    outputs = []
    hook = model.quantizer.register_forward_hook(
        lambda module, args, output: outputs.append(output)
    )
    loss, stats, weight = model(
        speech,
        torch.tensor([10, 8, 9], device=device),
        torch.tensor([[2, 3], [3, 2], [2, 3]], device=device),
        torch.tensor([2, 2, 2], device=device),
        reference,
        torch.tensor([7, 0, 12], device=device),
        pair_mask[:, None] if column_mask else pair_mask,
    )
    hook.remove()
    assert outputs[0].quantized.size(0) == 2  # Only the two references.
    assert outputs[1].quantized.size(0) == 3  # All three ASR inputs.
    pc = torch.stack(
        [
            outputs[0].soft_probabilities[0, :7].mean(0),
            outputs[0].soft_probabilities[1, :12].mean(0),
        ]
    ).detach()
    pd = torch.stack(
        [
            outputs[1].soft_probabilities[0, :10].mean(0),
            outputs[1].soft_probabilities[2, :9].mean(0),
        ]
    )
    expected = (pc * (pc.log() - pd.log())).sum(-1).mean()
    torch.testing.assert_close(stats["loss_sir"].squeeze(), expected.detach())
    torch.testing.assert_close(
        loss, model.asr_weight * stats["loss_asr"] + 0.4 * expected
    )
    assert stats["sir_num_pairs"].item() == 2
    assert weight.item() == 3
    loss.backward()
    assert reference.grad is None
    assert torch.isfinite(speech.grad).all()
    assert speech.grad[[0, 2]].abs().sum() > 0
    if sir_only:
        assert speech.grad[1].eq(0).all()
    else:
        assert speech.grad[1].abs().sum() > 0
    assert model.frontend.projection.weight.grad.abs().sum() > 0
    assert model.quantizer.codebook.grad.abs().sum() > 0


@pytest.mark.parametrize("provide_references", [False, True])
def test_sir_zero_pairs_matches_asr_only_loss_and_gradients(provide_references):
    torch.manual_seed(3)
    model = _build_sir_model(sir_weight=0.4)
    baseline = copy.deepcopy(model)
    baseline.sir_weight = 0.0
    speech = torch.randn(3, 10, 6, requires_grad=True)
    baseline_speech = speech.detach().clone().requires_grad_()
    args = (
        torch.tensor([10, 8, 9]),
        torch.tensor([[2, 3], [3, 2], [2, 3]]),
        torch.tensor([2, 2, 2]),
    )
    kwargs = {"sir_pair_mask": torch.zeros(3, dtype=torch.bool)}
    if provide_references:
        kwargs.update(
            healthy_speech=torch.full((3, 12, 6), float("nan")),
            healthy_speech_lengths=torch.zeros(3, dtype=torch.long),
        )
    outputs = []
    hook = model.quantizer.register_forward_hook(
        lambda module, args, output: outputs.append(output)
    )
    torch.manual_seed(7)
    loss, stats, weight = model(speech, *args, **kwargs)
    hook.remove()
    torch.manual_seed(7)
    baseline_loss, _, _ = baseline(baseline_speech, *args)
    torch.testing.assert_close(loss, baseline_loss)
    assert stats["loss_sir"].item() == 0.0
    assert stats["sir_num_pairs"].item() == 0
    assert weight.item() == 3
    assert len(outputs) == 1
    assert outputs[0].soft_probabilities is None
    assert model.quantizer.num_updates.item() == 1
    loss.backward()
    baseline_loss.backward()
    torch.testing.assert_close(speech.grad, baseline_speech.grad)
    assert speech.grad.flatten(1).abs().sum(1).gt(0).all()
    for parameter, baseline_parameter in zip(model.parameters(), baseline.parameters()):
        torch.testing.assert_close(parameter.grad, baseline_parameter.grad)


@pytest.mark.parametrize("sample_gumbel", [False, True])
def test_sir_all_pairs_mask_matches_legacy_healthy_reference_api(sample_gumbel):
    model = _build_sir_model(sir_weight=0.4, quantizer_sample_gumbel=sample_gumbel)
    args = (
        torch.randn(2, 10, 6),
        torch.tensor([10, 8]),
        torch.tensor([[2, 3], [3, 2]]),
        torch.tensor([2, 2]),
        torch.randn(2, 12, 6),
        torch.tensor([7, 12]),
    )
    torch.manual_seed(7)
    loss, stats, _ = model(*args)
    model.quantizer.set_num_updates(0)
    torch.manual_seed(7)
    masked_loss, masked_stats, _ = model(
        *args, sir_pair_mask=torch.ones(2, dtype=torch.bool)
    )
    torch.testing.assert_close(loss, masked_loss)
    torch.testing.assert_close(stats["loss_sir"], masked_stats["loss_sir"])
    assert masked_stats["sir_num_pairs"].item() == 2


@pytest.mark.parametrize(
    "pair_mask, message",
    [
        (torch.ones(2, 2), "must have shape"),
        (torch.ones(3), "must have shape"),
        (torch.tensor([1, -1]), "only 0 or 1"),
        (torch.tensor([1.0, 0.5]), "only 0 or 1"),
        (torch.tensor([1.0, float("nan")]), "only 0 or 1"),
        (torch.ones(2, device="meta"), "same device"),
    ],
)
def test_sir_rejects_invalid_pair_masks(pair_mask, message):
    model = _build_sir_model(sir_weight=0.1)
    with pytest.raises(ValueError, match=message):
        model(
            torch.randn(2, 10, 6),
            torch.tensor([10, 8]),
            torch.tensor([[2, 3], [3, 2]]),
            torch.tensor([2, 2]),
            sir_pair_mask=pair_mask,
        )


@pytest.mark.parametrize(
    "healthy, lengths, message",
    [
        (None, None, "required for selected SIR pairs"),
        (torch.randn(2, 10, 6), None, "provided together"),
        (None, torch.tensor([10, 8]), "provided together"),
        (torch.randn(1, 10, 6), torch.tensor([10]), "batch and feature"),
        (torch.randn(2, 10, 5), torch.tensor([10, 8]), "batch and feature"),
        (torch.randn(2, 10, 6), torch.tensor([[10, 8]]), "must have shape"),
        (torch.randn(2, 10, 6), torch.tensor([10]), "must have shape"),
        (torch.randn(2, 10, 6), torch.tensor([0, 8]), "valid positive"),
        (torch.randn(2, 10, 6), torch.tensor([11, 8]), "valid positive"),
        (
            torch.empty(2, 10, 6, device="meta"),
            torch.tensor([10, 8]),
            "same device",
        ),
    ],
)
def test_sir_rejects_missing_or_invalid_pairs(healthy, lengths, message):
    model = _build_sir_model(sir_weight=0.1)
    with pytest.raises(ValueError, match=message):
        model(
            torch.randn(2, 10, 6),
            torch.tensor([10, 8]),
            torch.tensor([[2, 3], [3, 2]]),
            torch.tensor([2, 2]),
            healthy,
            lengths,
        )


def test_sir_rejects_negative_weight():
    with pytest.raises(ValueError, match="weights must be non-negative"):
        _build_sir_model(sir_weight=-0.1)


@pytest.mark.parametrize("local_pairs", [0, 1, 3])
def test_sir_ddp_normalization_uses_global_pair_count(monkeypatch, local_pairs):
    model = _build_sir_model(sir_weight=0.1)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def all_reduce(counts):
        # The other worker has three pairs in a five-utterance batch.
        counts.add_(counts.new_tensor([3, 5]))

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    local_mean = torch.tensor(2.0, requires_grad=True)
    scaled, pair_count = model._normalize_sir_loss(local_mean, local_pairs, 4)
    expected_scale = local_pairs * 9 / (4 * (local_pairs + 3))
    torch.testing.assert_close(scaled, local_mean * expected_scale)
    assert pair_count.item() == local_pairs + 3
    scaled.backward()
    assert local_mean.grad.item() == pytest.approx(expected_scale)


@pytest.mark.parametrize("provide_references", [False, True])
def test_sir_rank_zero_evaluation_does_not_require_other_ranks(
    monkeypatch, provide_references
):
    model = _build_sir_model(sir_weight=0.1).eval()
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def all_reduce(*args, **kwargs):
        raise AssertionError("Other ranks do not participate in attention plotting")

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    batch = dict(
        speech=torch.randn(2, 10, 6),
        speech_lengths=torch.tensor([10, 8]),
        text=torch.tensor([[2, 3], [3, 2]]),
        text_lengths=torch.tensor([2, 2]),
    )
    if provide_references:
        batch.update(
            healthy_speech=torch.randn(2, 12, 6),
            healthy_speech_lengths=torch.tensor([12, 0]),
            sir_pair_mask=torch.tensor([1, 0]),
        )
    with torch.no_grad():
        loss, stats, weight = model(**batch)
    assert torch.isfinite(loss)
    assert stats["sir_num_pairs"].item() == int(provide_references)
    assert weight.item() == 2
    attentions = calculate_all_attentions(model, batch)
    assert attentions
    for samples in attentions.values():
        assert len(samples) == 2
        assert all(torch.isfinite(attention).all() for attention in samples)
    assert model.quantizer.num_updates.item() == 0


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
