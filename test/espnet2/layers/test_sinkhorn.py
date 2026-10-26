import pytest
import torch

from espnet2.layers.sinkhorn import UniformSinkhornDivergence


def test_uniform_sinkhorn_is_zero_for_identical_point_clouds():
    points = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    divergence = UniformSinkhornDivergence(epsilon=0.5, num_iterations=50)

    loss = divergence(points, points)

    assert loss.item() == pytest.approx(0.0, abs=1.0e-6)


def test_uniform_sinkhorn_backpropagates_to_source_only():
    source = torch.tensor([[0.5, 0.0], [2.5, 0.0], [0.5, 2.0]], requires_grad=True)
    reference = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    divergence = UniformSinkhornDivergence(epsilon=0.5, num_iterations=50)

    loss = divergence(source, reference)
    loss.backward()

    assert loss.item() > 0.0
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()
    assert reference.grad is None


def test_sinkhorn_accepts_explicit_cluster_masses():
    source = torch.tensor(
        [[0.5, 0.0], [2.5, 0.0], [0.5, 2.0]], requires_grad=True
    )
    reference = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    source_mass = torch.tensor([4.0, 1.0, 1.0], requires_grad=True)
    reference_mass = torch.tensor([2.0, 3.0, 1.0])
    divergence = UniformSinkhornDivergence(epsilon=0.5, num_iterations=100)

    loss = divergence(
        source,
        reference,
        source_mass=source_mass,
        reference_mass=reference_mass,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert source.grad is not None
    assert torch.isfinite(source.grad).all()
    assert source_mass.grad is not None
    assert torch.isfinite(source_mass.grad).all()
    assert not torch.isfinite(divergence.reference_self_cost)


def test_sinkhorn_validates_explicit_cluster_masses():
    points = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    divergence = UniformSinkhornDivergence()

    with pytest.raises(ValueError, match="source_mass must have shape"):
        divergence(points, points, source_mass=torch.ones(3))
    with pytest.raises(ValueError, match="values must all be positive"):
        divergence(points, points, source_mass=torch.tensor([1.0, 0.0]))
