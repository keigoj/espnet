import pytest
import torch

from espnet2.asr.frontend.token_embedding import TokenEmbedding


@pytest.mark.parametrize("use_positional_encoding", [False, True])
def test_ids_and_straight_through_assignments_share_embeddings(use_positional_encoding):
    """B=2 utterances, T=3 frames, K=3 clusters, E=4 embedding dimensions."""
    embedding = TokenEmbedding(
        input_size=3,
        embed_dim=4,
        use_positional_encoding=use_positional_encoding,
        positional_dropout_rate=0.0,
    )
    ids = torch.tensor([[0, 1, 2], [2, 0, 1]])  # (B, T).
    lengths = torch.tensor([3, 2])  # (B,).
    logits = torch.randn(2, 3, 3, requires_grad=True)  # (B, T, K).
    soft = logits.softmax(-1)  # Normalize over clusters K.
    hard = torch.nn.functional.one_hot(ids, num_classes=3).float()
    assignments = hard - soft.detach() + soft  # (B, T, K).
    from_ids, _ = embedding(ids, lengths)
    from_assignments, returned_lengths = embedding(assignments, lengths)
    torch.testing.assert_close(from_assignments, from_ids)
    assert from_assignments.shape == (2, 3, 4)
    assert from_assignments.device == logits.device
    assert torch.equal(returned_lengths, lengths)
    from_assignments.square().sum().backward()  # (B, T, E) -> scalar.
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0
    assert embedding.embed.weight.grad.abs().sum() > 0


def test_token_embedding_masks_padding_and_accepts_reduced_precision_assignments():
    """Zero padded assignments remain zero before positional encoding."""
    embedding = TokenEmbedding(3, 4, use_positional_encoding=False)
    assignments = torch.tensor([[[1, 0, 0], [0, 0, 0]]], dtype=torch.bfloat16)
    output, lengths = embedding(assignments, torch.tensor([1]))
    assert output.dtype == embedding.embed.weight.dtype
    assert output.shape == (1, 2, 4)
    assert output[0, 1].eq(0).all()
    assert lengths.tolist() == [1]
