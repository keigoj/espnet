import numpy as np
import pytest
import torch

from espnet2.layers.gumbel_kmeans import GumbelKMeans


def test_gumbel_kmeans_straight_through_backward():
    quantizer = GumbelKMeans(
        input_dim=4,
        num_clusters=3,
        temperature=2.0,
        min_temperature=0.5,
        temperature_decay=0.9,
        hard=True,
    )
    inputs = torch.randn(2, 5, 4, requires_grad=True)
    lengths = torch.tensor([5, 3], dtype=torch.long)

    output = quantizer(inputs, lengths)
    loss = (
        output.quantized.square().mean()
        + output.distortion_loss
        + output.diversity_loss
    )
    loss.backward()

    assert output.quantized.shape == inputs.shape
    assert output.indices.shape == inputs.shape[:2]
    assert output.probabilities.shape == (2, 5, 3)
    assert output.indices[1, 3:].eq(-1).all()
    assert output.probabilities[1, 3:].eq(0).all()
    assert output.quantized[1, 3:].eq(0).all()
    assert inputs.grad is not None
    assert quantizer.codebook.grad is not None
    assert quantizer.current_temperature == pytest.approx(1.8)


def test_gumbel_kmeans_eval_is_deterministic():
    quantizer = GumbelKMeans(input_dim=2, num_clusters=2, hard=True)
    quantizer.set_codebook(torch.tensor([[0.0, 0.0], [2.0, 2.0]]))
    quantizer.eval()
    inputs = torch.tensor([[[0.1, 0.2], [1.9, 2.1]]])
    lengths = torch.tensor([2], dtype=torch.long)

    first = quantizer(inputs, lengths)
    second = quantizer(inputs, lengths)

    assert torch.equal(first.indices, torch.tensor([[0, 1]]))
    assert torch.equal(first.indices, second.indices)
    assert torch.equal(first.quantized, second.quantized)
    assert quantizer.num_updates.item() == 0


def test_gumbel_kmeans_temperature_pauses_while_codebook_is_frozen():
    quantizer = GumbelKMeans(
        input_dim=2, num_clusters=3, temperature=2.0, temperature_decay=0.9
    ).train()
    inputs = torch.randn(2, 5, 2)
    lengths = torch.tensor([5, 3])

    quantizer.codebook.requires_grad_(False)
    for _ in range(3):
        output = quantizer(inputs, lengths)
        assert output.temperature.item() == pytest.approx(2.0)
        assert quantizer.num_updates.item() == 0

    quantizer.codebook.requires_grad_(True)
    output = quantizer(inputs, lengths)
    assert output.temperature.item() == pytest.approx(2.0)
    assert quantizer.num_updates.item() == 1
    assert quantizer.current_temperature == pytest.approx(1.8)

    # A later freeze pauses the existing schedule rather than resetting it.
    quantizer.codebook.requires_grad_(False)
    output = quantizer(inputs, lengths)
    assert output.temperature.item() == pytest.approx(1.8)
    assert quantizer.num_updates.item() == 1
    quantizer.codebook.requires_grad_(True)
    output = quantizer(inputs, lengths)
    assert output.temperature.item() == pytest.approx(1.8)
    assert quantizer.num_updates.item() == 2


def test_gumbel_kmeans_deterministic_training_preserves_upstream_gradients():
    quantizer = GumbelKMeans(
        input_dim=1, num_clusters=2, temperature=1.0, hard=True
    ).train()
    quantizer.set_codebook(torch.tensor([[0.0], [2.0]]))
    inputs = torch.tensor([[[0.5], [1.5], [100.0]]], requires_grad=True)
    lengths = torch.tensor([2], dtype=torch.long)

    output = quantizer(inputs, lengths, sample_gumbel=False)
    expected = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]])
    assert torch.equal(output.probabilities, expected)
    assert torch.equal(output.indices, torch.tensor([[0, 1, -1]]))
    assert torch.equal(output.quantized, torch.tensor([[[0.0], [2.0], [0.0]]]))
    assert quantizer.num_updates.item() == 0

    # Only the downstream representation is supervised; auxiliary losses must
    # not hide a broken assignment gradient to the SSL frontend.
    output.quantized.sum().backward()
    assert inputs.grad is not None
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad[0, :2].gt(0).all()
    assert inputs.grad[0, 2].eq(0).all()
    assert quantizer.codebook.grad is not None
    assert torch.isfinite(quantizer.codebook.grad).all()
    assert quantizer.codebook.grad.abs().sum() > 0

    quantizer.eval()
    evaluated = quantizer(inputs.detach(), lengths)
    assert torch.equal(output.quantized.detach(), evaluated.quantized)


def test_gumbel_kmeans_amp_preserves_nearby_centroid_distances():
    quantizer = GumbelKMeans(input_dim=2, num_clusters=2, hard=True).eval()
    quantizer.set_codebook(torch.tensor([[100.0, 100.0], [101.0, 101.0]]))
    inputs = torch.tensor([[[100.75, 100.75]]], requires_grad=True)
    lengths = torch.tensor([1], dtype=torch.long)
    expected = quantizer._distances(inputs)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        distances = quantizer._distances(inputs)
        output = quantizer(inputs, lengths)

    torch.testing.assert_close(distances, expected, rtol=0, atol=0)
    assert output.indices.item() == 1
    output.distortion_loss.backward()
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(quantizer.codebook.grad).all()


def test_gumbel_kmeans_logit_scale_controls_softness_without_changing_argmax():
    inputs = torch.tensor([[[0.25], [1.75]]])
    lengths = torch.tensor([2], dtype=torch.long)
    centers = torch.tensor([[0.0], [2.0]])
    unscaled = GumbelKMeans(
        input_dim=1,
        num_clusters=2,
        temperature=1.0,
        hard=False,
        logit_scale=1.0,
    )
    tempered = GumbelKMeans(
        input_dim=1,
        num_clusters=2,
        temperature=1.0,
        hard=False,
        logit_scale=0.1,
    )
    unscaled.set_codebook(centers)
    tempered.set_codebook(centers)
    unscaled.eval()
    tempered.eval()

    unscaled_output = unscaled(inputs, lengths)
    tempered_output = tempered(inputs, lengths)

    assert torch.equal(unscaled_output.indices, tempered_output.indices)
    assert torch.all(
        unscaled_output.probabilities.max(dim=-1).values
        > tempered_output.probabilities.max(dim=-1).values
    )


@pytest.mark.parametrize("logit_scale", [0.0, -1.0])
def test_gumbel_kmeans_rejects_non_positive_logit_scale(logit_scale):
    with pytest.raises(ValueError, match="logit_scale must be positive"):
        GumbelKMeans(input_dim=2, num_clusters=2, logit_scale=logit_scale)


def test_gumbel_kmeans_validates_codebook_shape():
    quantizer = GumbelKMeans(input_dim=2, num_clusters=3)
    with pytest.raises(ValueError, match="Expected centers"):
        quantizer.set_codebook(torch.randn(2, 2))


def test_gumbel_kmeans_has_future_ot_mass_mode_placeholder():
    quantizer = GumbelKMeans(input_dim=2, num_clusters=2)
    assert quantizer.ot_mass_mode == "uniform"

    with pytest.raises(NotImplementedError, match="ot_mass_mode='uniform'"):
        GumbelKMeans(input_dim=2, num_clusters=2, ot_mass_mode="cluster_usage")


def test_gumbel_kmeans_losses_are_dimension_normalized_and_masked():
    quantizer = GumbelKMeans(input_dim=4, num_clusters=2, hard=True)
    quantizer.set_codebook(torch.zeros(2, 4))
    quantizer.eval()
    inputs = torch.tensor([[[1.0] * 4, [100.0] * 4]])
    lengths = torch.tensor([1], dtype=torch.long)

    output = quantizer(inputs, lengths)

    assert output.distortion_loss.item() == pytest.approx(1.0)
    assert output.commitment_loss.item() == pytest.approx(1.0)


def test_gumbel_kmeans_distortion_uses_quantized_reconstruction():
    quantizer = GumbelKMeans(
        input_dim=1,
        num_clusters=2,
        temperature=1.0,
        hard=False,
    )
    quantizer.set_codebook(torch.tensor([[0.0], [2.0]]))
    quantizer.eval()
    inputs = torch.tensor([[[0.5]]])
    lengths = torch.tensor([1], dtype=torch.long)

    output = quantizer(inputs, lengths)
    expected = (inputs - output.quantized).square().mean()

    assert output.distortion_loss.item() == pytest.approx(expected.item())


def test_gumbel_kmeans_loads_fixed_reference_and_initializes_codebook(tmp_path):
    centers = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    path = tmp_path / "healthy_centroids.npy"
    np.save(path, centers)

    quantizer = GumbelKMeans(
        input_dim=2,
        num_clusters=2,
        reference_centroid_path=str(path),
        initialize_from_reference=True,
        ot_epsilon=0.5,
        ot_iterations=50,
    )

    assert torch.equal(quantizer.codebook.detach(), torch.from_numpy(centers))
    assert torch.equal(quantizer.reference_codebook, torch.from_numpy(centers))
    assert not quantizer.reference_codebook.requires_grad
    assert "reference_codebook" in quantizer.state_dict()

    with torch.no_grad():
        quantizer.codebook.add_(0.5)
    loss = quantizer.optimal_transport_loss()
    loss.backward()

    assert loss.item() > 0.0
    assert quantizer.codebook.grad is not None

    quantizer.codebook.grad = None
    source_mass = torch.tensor([3.0, 1.0], requires_grad=True)
    weighted_loss = quantizer.optimal_transport_loss(
        source_mass=source_mass,
        reference_mass=torch.tensor([1.0, 2.0]),
    )
    weighted_loss.backward()

    assert torch.isfinite(weighted_loss)
    assert quantizer.codebook.grad is not None
    assert source_mass.grad is not None


def test_gumbel_kmeans_uses_separate_initial_and_reference_centroids(tmp_path):
    initial = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    reference = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    initial_path = tmp_path / "sap_centroids.npy"
    reference_path = tmp_path / "healthy_centroids.npy"
    np.save(initial_path, initial)
    np.save(reference_path, reference)

    quantizer = GumbelKMeans(
        input_dim=2,
        num_clusters=2,
        initial_centroid_path=str(initial_path),
        reference_centroid_path=str(reference_path),
        initialize_from_reference=True,
    )

    assert torch.equal(quantizer.codebook.detach(), torch.from_numpy(initial))
    assert torch.equal(quantizer.initial_codebook, torch.from_numpy(initial))
    assert torch.equal(quantizer.reference_codebook, torch.from_numpy(reference))
    assert "initial_codebook" not in quantizer.state_dict()
    assert "reference_codebook" in quantizer.state_dict()

    with torch.no_grad():
        quantizer.codebook.zero_()
    quantizer.espnet_initialization_fn()
    assert torch.equal(quantizer.codebook.detach(), torch.from_numpy(initial))
