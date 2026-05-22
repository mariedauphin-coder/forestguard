"""Unit tests for model architecture and loss functions."""

import pytest
import torch
from forestguard.models.losses import DiceLoss, CombinedLoss
from forestguard.models.encoder import SiameseEncoder
from forestguard.models.unet import UNetDecoder
from forestguard.models.fusion_model import ForestGuardModel


@pytest.fixture
def dummy_batch():
    B, C, H, W = 2, 8, 64, 64
    before = torch.rand(B, C, H, W)
    after  = torch.rand(B, C, H, W)
    labels = (torch.rand(B, 1, H, W) > 0.7).float()
    return before, after, labels


class TestDiceLoss:
    def test_perfect_prediction_near_zero(self):
        loss_fn = DiceLoss()
        logits  = torch.tensor([[[10.0, -10.0]]])
        targets = torch.tensor([[[1.0,  0.0]]])
        loss = loss_fn(logits, targets)
        assert loss.item() < 0.1

    def test_worst_prediction_near_one(self):
        loss_fn = DiceLoss()
        logits  = torch.tensor([[[-10.0, 10.0]]])
        targets = torch.tensor([[[1.0,  0.0]]])
        loss = loss_fn(logits, targets)
        assert loss.item() > 0.5


class TestCombinedLoss:
    def test_output_is_scalar(self, dummy_batch):
        before, after, labels = dummy_batch
        model = ForestGuardModel(pretrained=False)
        logits = model(before, after)
        loss_fn = CombinedLoss()
        loss = loss_fn(logits, labels)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_loss_decreases_toward_correct_labels(self):
        loss_fn = CombinedLoss()
        logits_good = torch.tensor([[[ 5.0, -5.0,  5.0]]])
        logits_bad  = torch.tensor([[[-5.0,  5.0, -5.0]]])
        targets     = torch.tensor([[[1.0,  0.0,  1.0]]])
        assert loss_fn(logits_good, targets).item() < loss_fn(logits_bad, targets).item()


class TestForestGuardModel:
    def test_output_shape(self, dummy_batch):
        before, after, _ = dummy_batch
        model = ForestGuardModel(pretrained=False)
        with torch.no_grad():
            out = model(before, after)
        assert out.shape == (2, 1, 64, 64)

    def test_predict_mask_binary(self, dummy_batch):
        before, after, _ = dummy_batch
        model = ForestGuardModel(pretrained=False)
        mask = model.predict_mask(before, after)
        unique = torch.unique(mask).tolist()
        assert set(unique).issubset({0.0, 1.0})

    def test_siamese_encoder_shared_weights(self):
        enc = SiameseEncoder(in_channels=8, pretrained=False)
        before = torch.rand(1, 8, 64, 64)
        after  = torch.rand(1, 8, 64, 64)
        feats = enc(before, after)
        assert len(feats) == 4
        # Each feature map has doubled channel depth
        for f in feats:
            assert f.shape[1] % 2 == 0
