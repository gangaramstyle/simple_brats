import torch

from simple_brats.models import CrossModalEncoder, EncoderConfig, TargetModalityPredictor


def _small_encoder() -> CrossModalEncoder:
    return CrossModalEncoder(
        EncoderConfig(
            patch_shape=(4, 4, 1),
            embed_dim=24,
            depth=2,
            num_heads=3,
            mlp_ratio=2.0,
            dropout=0.0,
        )
    ).eval()


def test_encoder_source_permutation_equivariance_and_common_shift_invariance() -> None:
    torch.manual_seed(20)
    encoder = _small_encoder()
    patches = torch.randn(2, 6, 4, 4, 1)
    modality_ids = torch.tensor([[0, 1, 2, 3, 0, 1], [3, 2, 1, 0, 3, 2]])
    coordinates = torch.randn(2, 6, 3) * 20
    anchor = torch.randn(2, 3) * 20

    reference = encoder(patches, modality_ids, coordinates, anchor)
    permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    permuted = encoder(
        patches[:, permutation],
        modality_ids[:, permutation],
        coordinates[:, permutation],
        anchor,
    )
    torch.testing.assert_close(permuted, reference[:, permutation], rtol=1e-5, atol=1e-6)

    shift = torch.tensor([[100.0, -40.0, 7.0], [-31.0, 12.0, 80.0]])
    shifted = encoder(patches, modality_ids, coordinates + shift[:, None], anchor + shift)
    torch.testing.assert_close(shifted, reference, rtol=1e-5, atol=1e-6)


def test_predictor_source_and_query_permutations_and_common_shift() -> None:
    torch.manual_seed(21)
    predictor = TargetModalityPredictor(
        embed_dim=24,
        output_dim=12,
        depth=1,
        num_heads=3,
        dropout=0.0,
    ).eval()
    source_tokens = torch.randn(2, 7, 24)
    source_coordinates = torch.randn(2, 7, 3) * 10
    query_coordinates = torch.randn(2, 4, 3) * 10
    target_modality_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    anchor = torch.randn(2, 3)

    reference = predictor(
        source_tokens,
        source_coordinates,
        query_coordinates,
        target_modality_ids,
        anchor,
    )

    source_permutation = torch.tensor([5, 2, 0, 6, 1, 4, 3])
    source_permuted = predictor(
        source_tokens[:, source_permutation],
        source_coordinates[:, source_permutation],
        query_coordinates,
        target_modality_ids,
        anchor,
    )
    torch.testing.assert_close(source_permuted, reference, rtol=1e-5, atol=1e-6)

    query_permutation = torch.tensor([2, 0, 3, 1])
    query_permuted = predictor(
        source_tokens,
        source_coordinates,
        query_coordinates[:, query_permutation],
        target_modality_ids[:, query_permutation],
        anchor,
    )
    torch.testing.assert_close(
        query_permuted, reference[:, query_permutation], rtol=1e-5, atol=1e-6
    )

    shift = torch.tensor([[25.0, -10.0, 100.0], [-30.0, 4.0, 8.0]])
    shifted = predictor(
        source_tokens,
        source_coordinates + shift[:, None],
        query_coordinates + shift[:, None],
        target_modality_ids,
        anchor + shift,
    )
    torch.testing.assert_close(shifted, reference, rtol=1e-5, atol=1e-6)
