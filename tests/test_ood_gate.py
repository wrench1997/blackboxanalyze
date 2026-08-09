import torch

from app.ood_gate import fit_ood_reference, nearest_reference_distances, ood_flags


def test_ood_gate_accepts_training_like_features_and_rejects_far_surface():
    reference = torch.tensor([[0.0, 0.0], [0.0, 0.2], [1.0, 1.0]], dtype=torch.float32)
    fit = fit_ood_reference(reference, quantile=0.95, slack=1.25)
    distances = nearest_reference_distances(torch.tensor([[0.0, 0.1], [100.0, 100.0]]), reference)
    flags = ood_flags(distances, fit)
    assert flags == [False, True]
    assert fit["threshold"] > 0
