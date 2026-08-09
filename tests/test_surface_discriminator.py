import torch

from app.surface_discriminator import SurfaceDiscriminator


def test_surface_discriminator_has_separate_surface_and_context_path():
    model = SurfaceDiscriminator()
    logits = model(torch.zeros(3, 256))
    assert logits.shape == (3, 7)
    assert torch.isfinite(logits).all()
