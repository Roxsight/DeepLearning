import torch
import torch.nn.functional as F
from src.model import FusionModel, cons_loss, get_loss


def make_model(use_gate=True):
    return FusionModel(backbone='vit_tiny_patch16_224', pretrained=False, use_gate=use_gate)


def make_batch(b=4):
    return torch.randn(b, 16, 3, 224, 224), torch.randn(b, 128, 200)


def test_shapes():
    model = make_model()
    frames, mel = make_batch()
    out = model(frames, mel)
    assert out['logit'].shape == (4,)
    assert out['u'].shape == (4, 16)
    assert out['g'].shape == (4, 16)
    assert out['u'].min() >= 0 and out['u'].max() <= 1
    assert out['g'].min() > 0 and out['g'].max() < 1


def test_backward():
    model = make_model()
    frames, mel = make_batch()
    y = torch.tensor([0., 1., 0., 1.])
    loss, parts = get_loss(model(frames, mel), y)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.alpha.grad is not None
    assert model.beta.grad is not None
    assert any(p.grad is not None for p in model.video.backbone.parameters())
    assert any(p.grad is not None for p in model.audio.parameters())
    assert model.proj_v.weight.grad is not None


def test_cons_loss_aligned_is_lower():
    pv = F.normalize(torch.randn(3, 16, 32), dim=-1)
    pa_random = F.normalize(torch.randn(3, 16, 32), dim=-1)
    y = torch.zeros(3)
    assert cons_loss(pv, pv.clone(), y) < cons_loss(pv, pa_random, y)


def test_cons_loss_ignores_fakes():
    pv = F.normalize(torch.randn(2, 16, 32), dim=-1)
    assert cons_loss(pv, pv, torch.ones(2)).item() == 0.0


def test_no_gate():
    model = make_model(use_gate=False)
    frames, mel = make_batch(2)
    assert model(frames, mel)['g'].min() == 1.0
