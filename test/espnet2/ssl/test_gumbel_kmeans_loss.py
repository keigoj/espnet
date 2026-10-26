import numpy as np
import pytest
import torch

from espnet2.ssl.loss.gumbel_kmeans import GumbelKMeansLoss


def test_gumbel_kmeans_ssl_loss_backward_and_extract():
    loss_function = GumbelKMeansLoss(
        encoder_output_size=4,
        num_clusters=3,
        kmeans_weight=1.0,
        diversity_weight=0.1,
    )
    encoder_output = [torch.randn(2, 5, 4, requires_grad=True)]
    lengths = torch.tensor([5, 3], dtype=torch.long)

    loss, stats = loss_function(encoder_output, lengths)
    loss.backward()

    assert encoder_output[0].grad is not None
    assert loss_function.quantizer.codebook.grad is not None
    assert "gumbel_kmeans_perplexity" in stats
    assert "gumbel_kmeans_variance" in stats

    loss_function.eval()
    units, unit_lengths = loss_function.extract_units(
        encoder_output[0].detach(), encoder_output, lengths
    )
    assert units.shape == (2, 5)
    assert torch.equal(unit_lengths, lengths)
    assert units[1, 3:].eq(-1).all()


def test_gumbel_kmeans_variance_regularizer_penalizes_collapse():
    loss_function = GumbelKMeansLoss(
        encoder_output_size=4,
        num_clusters=3,
        kmeans_weight=0.0,
        diversity_weight=0.0,
        variance_weight=1.0,
    )
    encoder_output = [torch.zeros(2, 5, 4, requires_grad=True)]
    lengths = torch.tensor([5, 3], dtype=torch.long)

    loss, stats = loss_function(encoder_output, lengths)
    loss.backward()

    assert stats["gumbel_kmeans_variance"] > 0.9
    assert encoder_output[0].grad is not None


def test_gumbel_kmeans_uniform_ot_loss(tmp_path):
    centers = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    path = tmp_path / "healthy_centroids.npy"
    np.save(path, centers)
    loss_function = GumbelKMeansLoss(
        encoder_output_size=2,
        num_clusters=3,
        kmeans_weight=0.0,
        diversity_weight=0.0,
        ot_weight=1.0,
        reference_centroid_path=str(path),
        ot_epsilon=0.5,
        ot_iterations=50,
    )
    with torch.no_grad():
        loss_function.quantizer.codebook.add_(0.5)
    encoder_output = [torch.randn(2, 5, 2)]
    lengths = torch.tensor([5, 3], dtype=torch.long)

    loss, stats = loss_function(encoder_output, lengths)
    loss.backward()

    assert stats["gumbel_kmeans_ot"] > 0.0
    assert loss_function.quantizer.codebook.grad is not None
    assert loss_function.quantizer.reference_codebook.grad is None


def test_gumbel_kmeans_ot_requires_reference():
    with pytest.raises(ValueError, match="reference_centroid_path"):
        GumbelKMeansLoss(
            encoder_output_size=2,
            num_clusters=3,
            ot_weight=1.0,
        )


def test_gumbel_kmeans_ssl_loss_uses_initial_centroids(tmp_path):
    initial = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    reference = np.array([[0.0, 0.0], [2.0, 2.0]], dtype=np.float32)
    initial_path = tmp_path / "sap_centroids.npy"
    reference_path = tmp_path / "healthy_centroids.npy"
    np.save(initial_path, initial)
    np.save(reference_path, reference)

    loss_function = GumbelKMeansLoss(
        encoder_output_size=2,
        num_clusters=2,
        logit_scale=0.25,
        initial_centroid_path=str(initial_path),
        reference_centroid_path=str(reference_path),
    )

    assert torch.equal(
        loss_function.quantizer.codebook.detach(), torch.from_numpy(initial)
    )
    assert torch.equal(
        loss_function.quantizer.reference_codebook, torch.from_numpy(reference)
    )
    assert loss_function.quantizer.logit_scale == pytest.approx(0.25)
