import io
from types import SimpleNamespace

import pytest
import torch

from espnet2.layers.gumbel_kmeans import GumbelKMeans
from espnet2.train.trainer import Trainer


class _StagedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frontend = torch.nn.Linear(2, 2)
        self.quantizer = torch.nn.Linear(2, 2)
        self.encoder = torch.nn.Linear(2, 2)

    def forward(self, inputs):
        return self.encoder(self.quantizer(self.frontend(inputs)))


def _options(prefixes, until_epoch):
    return SimpleNamespace(
        staged_freeze_param=prefixes,
        staged_freeze_until_epoch=until_epoch,
    )


def test_staged_freeze_changes_state_at_epoch_boundary():
    model = _StagedModel()
    options = _options(["frontend", "quantizer"], 2)
    Trainer._validate_staged_freeze_options(model, options)

    matched, changed = Trainer._set_staged_freeze_state(model, 1, options)
    assert matched == 4
    assert changed == 4
    assert not any(parameter.requires_grad for parameter in model.frontend.parameters())
    assert not any(
        parameter.requires_grad for parameter in model.quantizer.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())

    matched, changed = Trainer._set_staged_freeze_state(model, 3, options)
    assert matched == 4
    assert changed == 4
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_staged_freeze_keeps_one_optimizer():
    model = _StagedModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    options = _options(["frontend"], 1)

    Trainer._set_staged_freeze_state(model, 1, options)
    model(torch.randn(2, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    encoder_step = optimizer.state[model.encoder.weight]["step"].item()
    assert model.frontend.weight not in optimizer.state

    Trainer._set_staged_freeze_state(model, 2, options)
    model(torch.randn(2, 2)).sum().backward()
    optimizer.step()
    assert optimizer.state[model.encoder.weight]["step"].item() == encoder_step + 1
    assert optimizer.state[model.frontend.weight]["step"].item() == 1


def test_staged_freeze_preserves_permanently_frozen_parameter():
    model = _StagedModel()
    model.frontend.bias.requires_grad_(False)
    options = _options(["frontend"], 1)

    Trainer._set_staged_freeze_state(model, 1, options)
    Trainer._set_staged_freeze_state(model, 2, options)

    assert model.frontend.weight.requires_grad
    assert not model.frontend.bias.requires_grad


def test_staged_freeze_rejects_unknown_prefix():
    model = _StagedModel()
    with pytest.raises(ValueError, match="does not match any parameter"):
        Trainer._validate_staged_freeze_options(model, _options(["missing"], 2))


@pytest.mark.parametrize("checkpoint_epoch", [None, 30, 31])
def test_staged_freeze_starts_temperature_decay_at_epoch31(checkpoint_epoch):
    def build_model():
        model = _StagedModel()
        model.quantizer = GumbelKMeans(
            input_dim=2, num_clusters=3, temperature=2.0, temperature_decay=0.9
        )
        return model.train()

    model = build_model()
    options = _options(["frontend", "quantizer"], 30)
    inputs = torch.randn(2, 5, 2)
    lengths = torch.tensor([5, 3])

    for epoch, expected_updates in [(1, 0), (30, 0), (31, 1), (32, 2)]:
        Trainer._set_staged_freeze_state(model, epoch, options)
        output = model.quantizer(model.frontend(inputs), lengths)
        model.encoder(output.quantized).sum().backward()

        assert model.encoder.weight.grad is not None
        assert torch.isfinite(model.encoder.weight.grad).all()
        if epoch <= 30:
            assert model.frontend.weight.grad is None
            assert model.quantizer.codebook.grad is None
        else:
            assert model.frontend.weight.grad is not None
            assert torch.isfinite(model.frontend.weight.grad).all()
            assert model.frontend.weight.grad.abs().sum() > 0
            assert model.quantizer.codebook.grad is not None
        assert model.quantizer.num_updates.item() == expected_updates
        expected_temperature = 1.8 if epoch == 32 else 2.0
        assert output.temperature.item() == pytest.approx(expected_temperature)
        model.zero_grad()

        if epoch == checkpoint_epoch:
            saved = io.BytesIO()
            torch.save(model.state_dict(), saved)
            saved.seek(0)
            model = build_model()
            model.load_state_dict(torch.load(saved, weights_only=True))
            assert model.quantizer.num_updates.item() == expected_updates
